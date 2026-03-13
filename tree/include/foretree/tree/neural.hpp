#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct NeuralLeafConfig {
    int    hidden_dim = 8;
    int    input_dim  = 0;
    double lr         = 0.01;
    int    epochs     = 50;
    double l2_reg     = 0.01;
    double clip_elem  = 10.0;
    double clip_global = 5.0;
    bool   cosine_lr  = true;
    double act_clip   = 50.0;
    double pred_sigma = 8.0;
    bool   enabled    = true;
    uint64_t seed     = 12345;

    // Tree compatibility (unused in neural predictor)
    int    min_samples          = 50;
    int    max_depth_start      = 3;
    double complexity_threshold = 0.1;
};

class NeuralLeafPredictor {
public:
    explicit NeuralLeafPredictor(const NeuralLeafConfig &cfg) 
        : P_(cfg.input_dim), H_(cfg.hidden_dim), cfg_(cfg) {
        if (P_ <= 0) throw std::runtime_error("NeuralLeafPredictor: input_dim must be positive");
        init_weights_();
        mu_.assign((size_t)P_, 0.0);
        inv_std_.assign((size_t)P_, 1.0);
        y_mean_ = 0.0;
        y_std_  = 1.0;
    }

    // Primary fit method with optional weights
    void fit(const double *X, int N, const double *y, const double *weights = nullptr) {
        if (N <= 0 || P_ <= 0) {
            valid_ = false;
            return;
        }
        if (!shapes_ok_()) {
            valid_ = false;
            return;
        }

        // Filter and standardize
        std::vector<int> keep;
        std::vector<double> keep_weights;
        if (!prepare_stats_(X, N, y, weights, keep, keep_weights)) {
            valid_ = false;
            return;
        }

        const int M = (int)keep.size();

        // Compact data
        std::vector<double> Xc((size_t)M * (size_t)P_);
        std::vector<double> yc((size_t)M);
        std::vector<double> wc((size_t)M);
        
        for (int t = 0; t < M; ++t) {
            const int i = keep[t];
            std::copy_n(X + (size_t)i * (size_t)P_, (size_t)P_, Xc.data() + (size_t)t * (size_t)P_);
            yc[(size_t)t] = y[i];
            wc[(size_t)t] = keep_weights[t];
        }

        // Normalize weights
        double wsum = std::accumulate(wc.begin(), wc.end(), 0.0);
        if (wsum > 1e-12) {
            const double wscale = (double)M / wsum;
            for (double &w : wc) w *= wscale;
        } else {
            std::fill(wc.begin(), wc.end(), 1.0);
        }

        // Train
        train_impl_(Xc, yc, wc, M);
    }

    // Convenience overloads
    void fit(const std::vector<std::vector<double>> &X, const std::vector<double> &y) {
        const int N = (int)X.size();
        if (N <= 0) {
            valid_ = false;
            return;
        }
        
        std::vector<double> X_flat((size_t)N * (size_t)P_);
        for (int i = 0; i < N; ++i) {
            if ((int)X[i].size() != P_) {
                std::cerr << "NeuralLeafPredictor: dimension mismatch at sample " << i 
                          << " got " << X[i].size() << " expected " << P_ << "\n";
                valid_ = false;
                return;
            }
            std::copy_n(X[i].data(), (size_t)P_, X_flat.data() + (size_t)i * (size_t)P_);
        }
        fit(X_flat.data(), N, y.data(), nullptr);
    }

    // Prediction
    double predict_one(const double *x_raw) const {
        if (!shapes_ok_()) return safe_bias_();
        
        std::vector<double> x((size_t)P_);
        normalize_row_(x_raw, x.data());

        std::vector<double> hidden((size_t)H_);
        forward_(x.data(), hidden.data());
        
        double pred_n = compute_output_(hidden.data());
        double pred = y_mean_ + y_std_ * pred_n;
        
        const double lo = y_mean_ - cfg_.pred_sigma * y_std_;
        const double hi = y_mean_ + cfg_.pred_sigma * y_std_;
        
        if (!std::isfinite(pred)) return safe_bias_();
        return std::clamp(pred, lo, hi);
    }

    double predict(const std::vector<double> &x_raw) const {
        if ((int)x_raw.size() != P_) return safe_bias_();
        return predict_one(x_raw.data());
    }

    int  input_dim() const { return P_; }
    int  hidden_dim() const { return H_; }
    bool valid() const { return valid_; }

private:
    int P_{0}, H_{0};
    NeuralLeafConfig cfg_{};

    std::vector<double> W1_, b1_, W2_;
    double b2_{0.0};

    std::vector<double> mW1_, vW1_, mW2_, vW2_, mb1_, vb1_;
    double mb2_{0.0}, vb2_{0.0};

    std::vector<double> mu_, inv_std_;
    double y_mean_{0.0}, y_std_{1.0};

    bool valid_{true};

    void init_weights_() {
        const size_t HP = (size_t)H_ * (size_t)P_;
        W1_.assign(HP, 0.0);
        b1_.assign((size_t)H_, 0.0);
        W2_.assign((size_t)H_, 0.0);
        b2_ = 0.0;

        std::mt19937_64 rng(cfg_.seed);
        const double scale = std::sqrt(2.0 / (double(P_) + double(H_)));
        std::normal_distribution<double> dist(0.0, scale);
        
        for (auto &w : W1_) w = dist(rng);
        for (auto &w : W2_) w = dist(rng);
    }

    bool shapes_ok_() const {
        return (P_ > 0) && 
               (int)W1_.size() == H_ * P_ && 
               (int)b1_.size() == H_ && 
               (int)W2_.size() == H_;
    }

    bool prepare_stats_(const double *X, int N, const double *y, const double *weights,
                       std::vector<int> &keep_idx, std::vector<double> &keep_weights) {
        keep_idx.clear();
        keep_weights.clear();
        keep_idx.reserve(N);
        keep_weights.reserve(N);

        for (int i = 0; i < N; ++i) {
            if (!std::isfinite(y[i])) continue;
            
            const double *row = X + (size_t)i * (size_t)P_;
            bool ok = true;
            for (int j = 0; j < P_; ++j) {
                if (!std::isfinite(row[j])) {
                    ok = false;
                    break;
                }
            }
            
            if (ok) {
                keep_idx.push_back(i);
                keep_weights.push_back(weights ? weights[i] : 1.0);
            }
        }

        const int M = (int)keep_idx.size();
        if (M < 8) return false;

        double wsum = std::accumulate(keep_weights.begin(), keep_weights.end(), 0.0);
        if (wsum < 1e-12) return false;

        // Weighted means
        std::fill(mu_.begin(), mu_.end(), 0.0);
        y_mean_ = 0.0;
        for (int k = 0; k < M; ++k) {
            const int i = keep_idx[k];
            const double w = keep_weights[k];
            const double *row = X + (size_t)i * (size_t)P_;
            for (int j = 0; j < P_; ++j) mu_[(size_t)j] += w * row[j];
            y_mean_ += w * y[i];
        }
        for (int j = 0; j < P_; ++j) mu_[(size_t)j] /= wsum;
        y_mean_ /= wsum;

        // Weighted variances
        std::vector<double> var((size_t)P_, 0.0);
        double y_var = 0.0;
        for (int k = 0; k < M; ++k) {
            const int i = keep_idx[k];
            const double w = keep_weights[k];
            const double *row = X + (size_t)i * (size_t)P_;
            for (int j = 0; j < P_; ++j) {
                const double d = row[j] - mu_[(size_t)j];
                var[(size_t)j] += w * d * d;
            }
            const double dy = y[i] - y_mean_;
            y_var += w * dy * dy;
        }
        
        for (int j = 0; j < P_; ++j) {
            const double s = std::sqrt(var[(size_t)j] / wsum + 1e-12);
            inv_std_[(size_t)j] = 1.0 / s;
        }
        y_std_ = std::sqrt(y_var / wsum + 1e-12);

        b2_ = 0.0;
        return true;
    }

    inline void normalize_row_(const double *x_raw, double *x_out) const {
        for (int j = 0; j < P_; ++j) {
            x_out[(size_t)j] = (x_raw[(size_t)j] - mu_[(size_t)j]) * inv_std_[(size_t)j];
        }
    }

    inline void forward_(const double *xnorm, double *hidden) const {
        for (int h = 0; h < H_; ++h) {
            double sum = b1_[(size_t)h];
            const size_t off = (size_t)h * (size_t)P_;
            for (int j = 0; j < P_; ++j) {
                sum += W1_[off + (size_t)j] * xnorm[(size_t)j];
            }
            sum = std::clamp(sum, -cfg_.act_clip, cfg_.act_clip);
            hidden[(size_t)h] = std::max(0.0, sum);
        }
    }

    inline double compute_output_(const double *hidden) const {
        double out = b2_;
        for (int h = 0; h < H_; ++h) {
            out += W2_[(size_t)h] * hidden[(size_t)h];
        }
        return out;
    }

    void train_impl_(const std::vector<double> &Xc, const std::vector<double> &yc, 
                    const std::vector<double> &wc, int M) {
        // Reset Adam state
        mW1_.assign(W1_.size(), 0.0);
        vW1_.assign(W1_.size(), 0.0);
        mW2_.assign(W2_.size(), 0.0);
        vW2_.assign(W2_.size(), 0.0);
        mb1_.assign(b1_.size(), 0.0);
        vb1_.assign(b1_.size(), 0.0);
        mb2_ = 0.0;
        vb2_ = 0.0;

        // Train/val split
        std::vector<int> ord(M);
        std::iota(ord.begin(), ord.end(), 0);
        std::mt19937_64 rng(cfg_.seed ^ 0xDEADBEEFULL);
        std::shuffle(ord.begin(), ord.end(), rng);

        if (M < 2) {
            valid_ = false;
            reset_to_baseline_();
            return;
        }

        const int Mtr_unclamped = static_cast<int>(std::round(0.8 * static_cast<double>(M)));
        const int Mtr = std::min(std::max(1, Mtr_unclamped), M - 1);

        auto row_ptr = [&](int t) { return Xc.data() + (size_t)ord[t] * (size_t)P_; };
        auto y_at = [&](int t) { return yc[(size_t)ord[t]]; };
        auto w_at = [&](int t) { return wc[(size_t)ord[t]]; };

        // Baseline
        double linear_val = compute_baseline_loss_(y_at, w_at, Mtr, M);

        // Training loop
        double best_val = std::numeric_limits<double>::infinity();
        int patience = 3, patience_ctr = 0;
        auto W1_best = W1_, W2_best = W2_, b1_best = b1_;
        double b2_best = b2_;

        const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
        double t_adam = 0.0;

        for (int epoch = 0; epoch < cfg_.epochs; ++epoch) {
            const double progress = (cfg_.epochs > 1) ? double(epoch) / double(cfg_.epochs - 1) : 1.0;
            const double lr = cfg_.lr * (cfg_.cosine_lr ? 
                (0.5 + 0.5 * std::cos(3.141592653589793 * progress)) : 1.0);

            // Compute gradients
            std::vector<double> gW1(W1_.size(), 0.0), gW2(W2_.size(), 0.0), gb1(b1_.size(), 0.0);
            double gb2 = 0.0;
            double wsum_tr = 0.0;

            compute_gradients_(row_ptr, y_at, w_at, 0, Mtr, gW1, gW2, gb1, gb2, wsum_tr);

            // Average and regularize
            const double inv_wsum = 1.0 / std::max(wsum_tr, 1e-12);
            for (size_t k = 0; k < gW1.size(); ++k) gW1[k] = gW1[k] * inv_wsum + cfg_.l2_reg * W1_[k];
            for (size_t k = 0; k < gW2.size(); ++k) gW2[k] = gW2[k] * inv_wsum + cfg_.l2_reg * W2_[k];
            for (size_t k = 0; k < gb1.size(); ++k) gb1[k] *= inv_wsum;
            gb2 *= inv_wsum;

            global_clip_(gW1, gW2, gb1, gb2, cfg_.clip_global);

            // Adam update
            t_adam += 1.0;
            const double bias1 = 1.0 - std::pow(beta1, t_adam);
            const double bias2 = 1.0 - std::pow(beta2, t_adam);

            adam_update_(W1_, mW1_, vW1_, gW1, lr, beta1, beta2, eps, bias1, bias2);
            adam_update_(W2_, mW2_, vW2_, gW2, lr, beta1, beta2, eps, bias1, bias2);
            adam_update_(b1_, mb1_, vb1_, gb1, lr, beta1, beta2, eps, bias1, bias2);

            mb2_ = beta1 * mb2_ + (1.0 - beta1) * gb2;
            vb2_ = beta2 * vb2_ + (1.0 - beta2) * (gb2 * gb2);
            const double mhat_b2 = mb2_ / bias1;
            const double vhat_b2 = vb2_ / bias2;
            b2_ -= lr * (mhat_b2 / (std::sqrt(vhat_b2) + eps));
            b2_ = std::clamp(b2_, -100.0, 100.0);

            // Validate
            const double cur_val = compute_val_loss_(row_ptr, y_at, w_at, Mtr, M);
            if (!std::isfinite(cur_val)) {
                valid_ = false;
                reset_to_baseline_();
                return;
            }

            if (cur_val < best_val) {
                best_val = cur_val;
                W1_best = W1_;
                W2_best = W2_;
                b1_best = b1_;
                b2_best = b2_;
                patience_ctr = 0;
            } else if (++patience_ctr >= patience) {
                W1_ = std::move(W1_best);
                W2_ = std::move(W2_best);
                b1_ = std::move(b1_best);
                b2_ = b2_best;
                break;
            }
        }

        // Never worse than linear
        if (!(best_val + 1e-9 < linear_val)) {
            valid_ = false;
            reset_to_baseline_();
        }
    }

    template<typename YAt, typename WAt>
    double compute_baseline_loss_(YAt y_at, WAt w_at, int start, int end) const {
        double loss = 0.0, wsum = 0.0;
        for (int t = start; t < end; ++t) {
            const double yn = (y_at(t) - y_mean_) / y_std_;
            const double w = w_at(t);
            loss += w * yn * yn; // constant 0 prediction in normalized space
            wsum += w;
        }
        return loss / std::max(wsum, 1e-12);
    }

    template<typename RowPtr, typename YAt, typename WAt>
    double compute_val_loss_(RowPtr row_ptr, YAt y_at, WAt w_at, int start, int end) const {
        double loss = 0.0, wsum = 0.0;
        std::vector<double> xnorm((size_t)P_);
        std::vector<double> hidden((size_t)H_);

        for (int t = start; t < end; ++t) {
            normalize_row_(row_ptr(t), xnorm.data());
            forward_(xnorm.data(), hidden.data());
            double pn = compute_output_(hidden.data());
            
            const double yn = (y_at(t) - y_mean_) / y_std_;
            const double e = pn - yn;
            const double w = w_at(t);
            loss += w * e * e;
            wsum += w;
        }
        return loss / std::max(wsum, 1e-12);
    }

    template<typename RowPtr, typename YAt, typename WAt>
    void compute_gradients_(RowPtr row_ptr, YAt y_at, WAt w_at, int start, int end,
                           std::vector<double> &gW1, std::vector<double> &gW2,
                           std::vector<double> &gb1, double &gb2, double &wsum_out) {
        std::vector<double> xnorm((size_t)P_);
        std::vector<double> hidden((size_t)H_);

        for (int t = start; t < end; ++t) {
            normalize_row_(row_ptr(t), xnorm.data());
            forward_(xnorm.data(), hidden.data());
            double pn = compute_output_(hidden.data());

            const double yn = (y_at(t) - y_mean_) / y_std_;
            const double err = pn - yn;
            const double w = w_at(t);
            wsum_out += w;

            const double g_out = std::clamp(w * err, -cfg_.clip_elem, cfg_.clip_elem);
            gb2 += g_out;
            for (int h = 0; h < H_; ++h) {
                gW2[(size_t)h] += g_out * hidden[(size_t)h];
            }

            for (int h = 0; h < H_; ++h) {
                if (hidden[(size_t)h] <= 0.0) continue;
                double g_h = std::clamp(g_out * W2_[(size_t)h], -cfg_.clip_elem, cfg_.clip_elem);
                gb1[(size_t)h] += g_h;
                const size_t off = (size_t)h * (size_t)P_;
                for (int j = 0; j < P_; ++j) {
                    gW1[off + (size_t)j] += g_h * xnorm[(size_t)j];
                }
            }
        }
    }

    void reset_to_baseline_() {
        std::fill(W1_.begin(), W1_.end(), 0.0);
        std::fill(W2_.begin(), W2_.end(), 0.0);
        std::fill(b1_.begin(), b1_.end(), 0.0);
        b2_ = 0.0;
    }

    static void global_clip_(std::vector<double> &gW1, std::vector<double> &gW2,
                            std::vector<double> &gb1, double &gb2, double max_norm) {
        if (max_norm <= 0.0) return;
        
        long double sumsq = 0.0L;
        for (double v : gW1) sumsq += (long double)v * v;
        for (double v : gW2) sumsq += (long double)v * v;
        for (double v : gb1) sumsq += (long double)v * v;
        sumsq += (long double)gb2 * gb2;

        const long double norm = std::sqrt(sumsq);
        if (norm > (long double)max_norm && norm > 0.0L) {
            const double scale = (double)max_norm / (double)norm;
            for (double &v : gW1) v *= scale;
            for (double &v : gW2) v *= scale;
            for (double &v : gb1) v *= scale;
            gb2 *= scale;
        }
    }

    static void adam_update_(std::vector<double> &w, std::vector<double> &m, std::vector<double> &v,
                            const std::vector<double> &g, double lr, double b1, double b2, 
                            double eps, double bias1, double bias2) {
        const size_t n = w.size();
        for (size_t i = 0; i < n; ++i) {
            const double gi = g[i];
            m[i] = b1 * m[i] + (1.0 - b1) * gi;
            v[i] = b2 * v[i] + (1.0 - b2) * (gi * gi);
            const double mhat = m[i] / bias1;
            const double vhat = v[i] / bias2;
            w[i] -= lr * (mhat / (std::sqrt(vhat) + eps));
            w[i] = std::clamp(w[i], -10.0, 10.0);
        }
    }

    inline double safe_bias_() const { return y_mean_; }
};
