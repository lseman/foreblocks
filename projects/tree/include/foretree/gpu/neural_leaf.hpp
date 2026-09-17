#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

enum class ActivationFn : uint8_t {
    ReLU = 0,
    Tanh = 1,
    Sigmoid = 2,
    GELU = 3
};

struct GpuNeuralLeafConfig {
    std::vector<int> layer_dims;  // e.g., {64, 32, 16} for 3 hidden layers
    ActivationFn activation = ActivationFn::ReLU;
    int input_dim = 0;
    double lr = 0.01;
    int epochs = 50;
    double l2_reg = 0.01;
    double clip_elem = 10.0;
    double clip_global = 5.0;
    bool cosine_lr = true;
    double act_clip = 50.0;
    double pred_sigma = 8.0;
    bool enabled = true;
    uint64_t seed = 12345;
    bool use_gpu = false;  // fallback to CPU if CUDA unavailable

    // Tree compatibility (unused)
    int min_samples = 50;
    int max_depth_start = 3;
    double complexity_threshold = 0.1;

    void validate() const {
        if (input_dim <= 0) throw std::runtime_error("GpuNeuralLeafConfig: input_dim must be positive");
        if (layer_dims.empty()) throw std::runtime_error("GpuNeuralLeafConfig: layer_dims must not be empty");
        for (int d : layer_dims) {
            if (d <= 0) throw std::runtime_error("GpuNeuralLeafConfig: all layer_dims must be positive");
        }
        if (lr <= 0.0) throw std::runtime_error("GpuNeuralLeafConfig: lr must be positive");
        if (epochs <= 0) throw std::runtime_error("GpuNeuralLeafConfig: epochs must be positive");
    }
};

class GpuNeuralLeafPredictor {
   public:
    explicit GpuNeuralLeafPredictor(const GpuNeuralLeafConfig& cfg)
        : cfg_(cfg) {
        cfg_.validate();
        build_architecture_();
        init_weights_();
        mu_.assign((size_t)cfg_.input_dim, 0.0);
        inv_std_.assign((size_t)cfg_.input_dim, 1.0);
        y_mean_ = 0.0;
        y_std_ = 1.0;
    }

    void fit(const double* X, int N, const double* y, const double* weights = nullptr) {
        if (N <= 0 || cfg_.input_dim <= 0) {
            valid_ = false;
            return;
        }
        if (!shapes_ok_()) {
            valid_ = false;
            return;
        }

        std::vector<int> keep;
        std::vector<double> keep_weights;
        if (!prepare_stats_(X, N, y, weights, keep, keep_weights)) {
            valid_ = false;
            return;
        }

        const int M = (int)keep.size();

        std::vector<double> Xc((size_t)M * (size_t)cfg_.input_dim);
        std::vector<double> yc((size_t)M);
        std::vector<double> wc((size_t)M);

        for (int t = 0; t < M; ++t) {
            const int i = keep[t];
            std::copy_n(X + (size_t)i * (size_t)cfg_.input_dim, (size_t)cfg_.input_dim,
                        Xc.data() + (size_t)t * (size_t)cfg_.input_dim);
            yc[(size_t)t] = y[i];
            wc[(size_t)t] = keep_weights[t];
        }

        double wsum = std::accumulate(wc.begin(), wc.end(), 0.0);
        if (wsum > 1e-12) {
            const double wscale = (double)M / wsum;
            for (double& w : wc) w *= wscale;
        } else {
            std::fill(wc.begin(), wc.end(), 1.0);
        }

        train_impl_(Xc, yc, wc, M);
    }

    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        const int N = (int)X.size();
        if (N <= 0) {
            valid_ = false;
            return;
        }

        std::vector<double> X_flat((size_t)N * (size_t)cfg_.input_dim);
        for (int i = 0; i < N; ++i) {
            if ((int)X[i].size() != cfg_.input_dim) {
                std::cerr << "GpuNeuralLeafPredictor: dimension mismatch at sample " << i
                          << " got " << X[i].size() << " expected " << cfg_.input_dim << "\n";
                valid_ = false;
                return;
            }
            std::copy_n(X[i].data(), (size_t)cfg_.input_dim,
                        X_flat.data() + (size_t)i * (size_t)cfg_.input_dim);
        }
        fit(X_flat.data(), N, y.data(), nullptr);
    }

    double predict_one(const double* x_raw) const {
        if (!shapes_ok_()) return safe_bias_();

        std::vector<double> x((size_t)cfg_.input_dim);
        normalize_row_(x_raw, x.data());

        std::vector<double> pred_n = forward_(x.data());
        double out = pred_n[0];
        double pred = y_mean_ + y_std_ * out;

        const double lo = y_mean_ - cfg_.pred_sigma * y_std_;
        const double hi = y_mean_ + cfg_.pred_sigma * y_std_;

        if (!std::isfinite(pred)) return safe_bias_();
        return std::clamp(pred, lo, hi);
    }

    double predict(const std::vector<double>& x_raw) const {
        if ((int)x_raw.size() != cfg_.input_dim) return safe_bias_();
        return predict_one(x_raw.data());
    }

    int input_dim() const { return cfg_.input_dim; }
    bool valid() const { return valid_; }

   private:
    GpuNeuralLeafConfig cfg_;
    std::vector<int> layer_dims_;  // [P, H1, H2, ..., 1]
    std::vector<std::vector<double>> weights_;  // weights_[i] = layer i weights
    std::vector<std::vector<double>> biases_;   // biases_[i] = layer i bias
    std::vector<std::vector<double>> m_weights_, v_weights_;
    std::vector<std::vector<double>> m_biases_, v_biases_;

    std::vector<double> mu_, inv_std_;
    double y_mean_{0.0}, y_std_{1.0};

    bool valid_{true};

    void build_architecture_() {
        layer_dims_.clear();
        layer_dims_.push_back(cfg_.input_dim);
        for (int h : cfg_.layer_dims) layer_dims_.push_back(h);
        layer_dims_.push_back(1);  // output layer

        const int num_layers = (int)layer_dims_.size() - 1;
        weights_.resize(num_layers);
        biases_.resize(num_layers);
        m_weights_.resize(num_layers);
        v_weights_.resize(num_layers);
        m_biases_.resize(num_layers);
        v_biases_.resize(num_layers);

        for (int i = 0; i < num_layers; ++i) {
            const int in_dim = layer_dims_[i];
            const int out_dim = layer_dims_[i + 1];
            weights_[i].assign((size_t)in_dim * (size_t)out_dim, 0.0);
            biases_[i].assign((size_t)out_dim, 0.0);
            m_weights_[i].assign((size_t)in_dim * (size_t)out_dim, 0.0);
            v_weights_[i].assign((size_t)in_dim * (size_t)out_dim, 0.0);
            m_biases_[i].assign((size_t)out_dim, 0.0);
            v_biases_[i].assign((size_t)out_dim, 0.0);
        }
    }

    void init_weights_() {
        std::mt19937_64 rng(cfg_.seed);
        const int num_layers = (int)layer_dims_.size() - 1;

        for (int i = 0; i < num_layers; ++i) {
            const int in_dim = layer_dims_[i];
            const int out_dim = layer_dims_[i + 1];
            const double scale = std::sqrt(2.0 / (double(in_dim) + double(out_dim)));
            std::normal_distribution<double> dist(0.0, scale);

            for (auto& w : weights_[i]) w = dist(rng);
            for (auto& b : biases_[i]) b = 0.0;
        }
    }

    bool shapes_ok_() const {
        const int num_layers = (int)layer_dims_.size() - 1;
        if ((int)weights_.size() != num_layers || (int)biases_.size() != num_layers)
            return false;
        for (int i = 0; i < num_layers; ++i) {
            const size_t expected_w = (size_t)layer_dims_[i] * (size_t)layer_dims_[i + 1];
            const size_t expected_b = (size_t)layer_dims_[i + 1];
            if ((int)weights_[i].size() != (int)expected_w || (int)biases_[i].size() != (int)expected_b)
                return false;
        }
        return true;
    }

    bool prepare_stats_(const double* X, int N, const double* y, const double* weights,
                        std::vector<int>& keep_idx, std::vector<double>& keep_weights) {
        keep_idx.clear();
        keep_weights.clear();
        keep_idx.reserve(N);
        keep_weights.reserve(N);

        for (int i = 0; i < N; ++i) {
            if (!std::isfinite(y[i])) continue;

            const double* row = X + (size_t)i * (size_t)cfg_.input_dim;
            bool ok = true;
            for (int j = 0; j < cfg_.input_dim; ++j) {
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

        std::fill(mu_.begin(), mu_.end(), 0.0);
        y_mean_ = 0.0;
        for (int k = 0; k < M; ++k) {
            const int i = keep_idx[k];
            const double w = keep_weights[k];
            const double* row = X + (size_t)i * (size_t)cfg_.input_dim;
            for (int j = 0; j < cfg_.input_dim; ++j) mu_[(size_t)j] += w * row[j];
            y_mean_ += w * y[i];
        }
        for (int j = 0; j < cfg_.input_dim; ++j) mu_[(size_t)j] /= wsum;
        y_mean_ /= wsum;

        std::vector<double> var((size_t)cfg_.input_dim, 0.0);
        double y_var = 0.0;
        for (int k = 0; k < M; ++k) {
            const int i = keep_idx[k];
            const double w = keep_weights[k];
            const double* row = X + (size_t)i * (size_t)cfg_.input_dim;
            for (int j = 0; j < cfg_.input_dim; ++j) {
                const double d = row[j] - mu_[(size_t)j];
                var[(size_t)j] += w * d * d;
            }
            const double dy = y[i] - y_mean_;
            y_var += w * dy * dy;
        }

        for (int j = 0; j < cfg_.input_dim; ++j) {
            const double s = std::sqrt(var[(size_t)j] / wsum + 1e-12);
            inv_std_[(size_t)j] = 1.0 / s;
        }
        y_std_ = std::sqrt(y_var / wsum + 1e-12);

        return true;
    }

    inline void normalize_row_(const double* x_raw, double* x_out) const {
        for (int j = 0; j < cfg_.input_dim; ++j) {
            x_out[(size_t)j] = (x_raw[(size_t)j] - mu_[(size_t)j]) * inv_std_[(size_t)j];
        }
    }

    double activate_(double x) const {
        x = std::clamp(x, -cfg_.act_clip, cfg_.act_clip);
        switch (cfg_.activation) {
            case ActivationFn::ReLU:
                return std::max(0.0, x);
            case ActivationFn::Tanh:
                return std::tanh(x);
            case ActivationFn::Sigmoid:
                return 1.0 / (1.0 + std::exp(-x));
            case ActivationFn::GELU:
                return x * 0.5 * (1.0 + std::tanh(0.79788456 * (x + 0.044715 * x * x * x)));
            default:
                return std::max(0.0, x);
        }
    }

    double activate_derivative_(double x, double activated) const {
        switch (cfg_.activation) {
            case ActivationFn::ReLU:
                return (x > 0.0) ? 1.0 : 0.0;
            case ActivationFn::Tanh: {
                const double t = activated;
                return 1.0 - t * t;
            }
            case ActivationFn::Sigmoid: {
                const double s = activated;
                return s * (1.0 - s);
            }
            case ActivationFn::GELU: {
                const double cdf = 0.5 * (1.0 + std::tanh(0.79788456 * (x + 0.044715 * x * x * x)));
                const double pdf = 0.39894228 * std::exp(-0.5 * x * x);
                return cdf + x * pdf;
            }
            default:
                return (x > 0.0) ? 1.0 : 0.0;
        }
    }

    std::vector<double> forward_(const double* xnorm) const {
        const int num_layers = (int)layer_dims_.size() - 1;
        std::vector<double> activations[2];
        activations[0] = std::vector<double>(xnorm, xnorm + cfg_.input_dim);

        for (int i = 0; i < num_layers; ++i) {
            const int in_dim = layer_dims_[i];
            const int out_dim = layer_dims_[i + 1];
            const bool is_last = (i == num_layers - 1);

            activations[(i + 1) % 2].assign((size_t)out_dim, 0.0);
            auto& out = activations[(i + 1) % 2];
            const auto& in = activations[i % 2];

            for (int o = 0; o < out_dim; ++o) {
                double sum = biases_[i][(size_t)o];
                const size_t off = (size_t)o * (size_t)in_dim;
                for (int j = 0; j < in_dim; ++j) {
                    sum += weights_[i][off + (size_t)j] * in[(size_t)j];
                }
                out[(size_t)o] = is_last ? sum : activate_(sum);
            }
        }

        return activations[num_layers % 2];
    }

    void train_impl_(const std::vector<double>& Xc, const std::vector<double>& yc,
                     const std::vector<double>& wc, int M) {
        for (int i = 0; i < (int)m_weights_.size(); ++i) {
            std::fill(m_weights_[i].begin(), m_weights_[i].end(), 0.0);
            std::fill(v_weights_[i].begin(), v_weights_[i].end(), 0.0);
            std::fill(m_biases_[i].begin(), m_biases_[i].end(), 0.0);
            std::fill(v_biases_[i].begin(), v_biases_[i].end(), 0.0);
        }

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

        auto row_ptr = [&](int t) { return Xc.data() + (size_t)ord[t] * (size_t)cfg_.input_dim; };
        auto y_at = [&](int t) { return yc[(size_t)ord[t]]; };
        auto w_at = [&](int t) { return wc[(size_t)ord[t]]; };

        double linear_val = compute_baseline_loss_(y_at, w_at, Mtr, M);

        double best_val = std::numeric_limits<double>::infinity();
        int patience = 3, patience_ctr = 0;
        auto weights_best = weights_;
        auto biases_best = biases_;

        const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
        double t_adam = 0.0;

        for (int epoch = 0; epoch < cfg_.epochs; ++epoch) {
            const double progress = (cfg_.epochs > 1) ? double(epoch) / double(cfg_.epochs - 1) : 1.0;
            const double lr = cfg_.lr * (cfg_.cosine_lr ? (0.5 + 0.5 * std::cos(3.141592653589793 * progress)) : 1.0);

            const int num_layers = (int)layer_dims_.size() - 1;
            std::vector<std::vector<double>> gW(num_layers), gb(num_layers);
            for (int i = 0; i < num_layers; ++i) {
                gW[i].assign(weights_[i].size(), 0.0);
                gb[i].assign(biases_[i].size(), 0.0);
            }
            double wsum_tr = 0.0;

            compute_gradients_(row_ptr, y_at, w_at, 0, Mtr, gW, gb, wsum_tr);

            const double inv_wsum = 1.0 / std::max(wsum_tr, 1e-12);
            for (int i = 0; i < num_layers; ++i) {
                for (size_t k = 0; k < gW[i].size(); ++k) gW[i][k] = gW[i][k] * inv_wsum + cfg_.l2_reg * weights_[i][k];
                for (size_t k = 0; k < gb[i].size(); ++k) gb[i][k] *= inv_wsum;
            }

            global_clip_(gW, gb, cfg_.clip_global);

            t_adam += 1.0;
            const double bias1 = 1.0 - std::pow(beta1, t_adam);
            const double bias2 = 1.0 - std::pow(beta2, t_adam);

            for (int i = 0; i < num_layers; ++i) {
                adam_update_(weights_[i], m_weights_[i], v_weights_[i], gW[i], lr, beta1, beta2, eps, bias1, bias2);
                adam_update_(biases_[i], m_biases_[i], v_biases_[i], gb[i], lr, beta1, beta2, eps, bias1, bias2);
            }

            const double cur_val = compute_val_loss_(row_ptr, y_at, w_at, Mtr, M);
            if (!std::isfinite(cur_val)) {
                valid_ = false;
                reset_to_baseline_();
                return;
            }

            if (cur_val < best_val) {
                best_val = cur_val;
                weights_best = weights_;
                biases_best = biases_;
                patience_ctr = 0;
            } else if (++patience_ctr >= patience) {
                weights_ = std::move(weights_best);
                biases_ = std::move(biases_best);
                break;
            }
        }

        if (!(best_val + 1e-9 < linear_val)) {
            valid_ = false;
            reset_to_baseline_();
        }
    }

    template <typename YAt, typename WAt>
    double compute_baseline_loss_(YAt y_at, WAt w_at, int start, int end) const {
        double loss = 0.0, wsum = 0.0;
        for (int t = start; t < end; ++t) {
            const double yn = (y_at(t) - y_mean_) / y_std_;
            const double w = w_at(t);
            loss += w * yn * yn;
            wsum += w;
        }
        return loss / std::max(wsum, 1e-12);
    }

    template <typename RowPtr, typename YAt, typename WAt>
    double compute_val_loss_(RowPtr row_ptr, YAt y_at, WAt w_at, int start, int end) const {
        double loss = 0.0, wsum = 0.0;
        std::vector<double> xnorm((size_t)cfg_.input_dim);

        for (int t = start; t < end; ++t) {
            normalize_row_(row_ptr(t), xnorm.data());
            auto pred_vec = forward_(xnorm.data());
            const double pn = pred_vec[0];

            const double yn = (y_at(t) - y_mean_) / y_std_;
            const double e = pn - yn;
            const double w = w_at(t);
            loss += w * e * e;
            wsum += w;
        }
        return loss / std::max(wsum, 1e-12);
    }

    template <typename RowPtr, typename YAt, typename WAt>
    void compute_gradients_(RowPtr row_ptr, YAt y_at, WAt w_at, int start, int end,
                            std::vector<std::vector<double>>& gW, std::vector<std::vector<double>>& gb,
                            double& wsum_out) {
        const int num_layers = (int)layer_dims_.size() - 1;
        std::vector<double> xnorm((size_t)cfg_.input_dim);
        std::vector<std::vector<double>> activations(num_layers + 1);

        for (int t = start; t < end; ++t) {
            normalize_row_(row_ptr(t), xnorm.data());
            activations[0] = std::vector<double>(xnorm.data(), xnorm.data() + cfg_.input_dim);

            for (int i = 0; i < num_layers; ++i) {
                const int in_dim = layer_dims_[i];
                const int out_dim = layer_dims_[i + 1];
                const bool is_last = (i == num_layers - 1);
                activations[i + 1].assign((size_t)out_dim, 0.0);

                for (int o = 0; o < out_dim; ++o) {
                    double sum = biases_[i][(size_t)o];
                    const size_t off = (size_t)o * (size_t)in_dim;
                    for (int j = 0; j < in_dim; ++j) {
                        sum += weights_[i][off + (size_t)j] * activations[i][(size_t)j];
                    }
                    activations[i + 1][(size_t)o] = is_last ? sum : activate_(sum);
                }
            }

            const double pn = activations[num_layers][0];
            const double yn = (y_at(t) - y_mean_) / y_std_;
            const double err = pn - yn;
            const double w = w_at(t);
            wsum_out += w;

            std::vector<double> delta(num_layers);
            delta[num_layers - 1] = std::clamp(w * err, -cfg_.clip_elem, cfg_.clip_elem);

            for (int i = num_layers - 1; i >= 0; --i) {
                const int in_dim = layer_dims_[i];
                const int out_dim = layer_dims_[i + 1];

                for (int o = 0; o < out_dim; ++o) {
                    gb[i][(size_t)o] += delta[i];
                    const size_t off = (size_t)o * (size_t)in_dim;
                    for (int j = 0; j < in_dim; ++j) {
                        gW[i][off + (size_t)j] += delta[i] * activations[i][(size_t)j];
                    }
                }

                if (i > 0) {
                    std::vector<double> next_delta(in_dim, 0.0);
                    for (int j = 0; j < in_dim; ++j) {
                        for (int o = 0; o < out_dim; ++o) {
                            const size_t off = (size_t)o * (size_t)in_dim;
                            next_delta[(size_t)j] += delta[i] * weights_[i][off + (size_t)j];
                        }
                        const double deriv = activate_derivative_(activations[i][(size_t)j], activations[i][(size_t)j]);
                        next_delta[(size_t)j] *= deriv;
                        next_delta[(size_t)j] = std::clamp(next_delta[(size_t)j], -cfg_.clip_elem, cfg_.clip_elem);
                    }
                    delta[i - 1] = next_delta[0];
                }
            }
        }
    }

    void reset_to_baseline_() {
        for (auto& w : weights_) std::fill(w.begin(), w.end(), 0.0);
        for (auto& b : biases_) std::fill(b.begin(), b.end(), 0.0);
    }

    static void global_clip_(std::vector<std::vector<double>>& gW, std::vector<std::vector<double>>& gb,
                             double max_norm) {
        if (max_norm <= 0.0) return;

        long double sumsq = 0.0L;
        for (const auto& g : gW)
            for (double v : g) sumsq += (long double)v * v;
        for (const auto& g : gb)
            for (double v : g) sumsq += (long double)v * v;

        const long double norm = std::sqrt(sumsq);
        if (norm > (long double)max_norm && norm > 0.0L) {
            const double scale = (double)max_norm / (double)norm;
            for (auto& g : gW)
                for (double& v : g) v *= scale;
            for (auto& g : gb)
                for (double& v : g) v *= scale;
        }
    }

    static void adam_update_(std::vector<double>& w, std::vector<double>& m, std::vector<double>& v,
                             const std::vector<double>& g, double lr, double b1, double b2, double eps, double bias1,
                             double bias2) {
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
