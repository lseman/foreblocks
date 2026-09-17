#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace foretree {

struct OrderedCategoricalConfig {
    bool enabled = false;
    std::vector<int> features;
    int permutations = 1;
    double prior = 0.0;
    double prior_weight = 1.0;
    uint64_t seed = 123456789ULL;
};

class OrderedCategoricalEncoder {
public:
    void reset() {
        input_features_ = 0;
        categorical_features_.clear();
        numerical_features_.clear();
        full_statistics_.clear();
        fitted_ = false;
    }

    std::vector<double> fit_transform(const double* X, int rows, int features, const double* target,
                                      OrderedCategoricalConfig config) {
        reset();
        validate_(X, rows, features, target, config);
        input_features_ = features;
        categorical_features_ = std::move(config.features);
        std::sort(categorical_features_.begin(), categorical_features_.end());
        categorical_features_.erase(std::unique(categorical_features_.begin(), categorical_features_.end()),
                                    categorical_features_.end());
        std::vector<uint8_t> categorical(static_cast<size_t>(features), 0);
        for (int feature : categorical_features_) {
            if (feature < 0 || feature >= features)
                throw std::invalid_argument("OrderedCategoricalEncoder: feature index out of range");
            categorical[static_cast<size_t>(feature)] = 1;
        }
        for (int feature = 0; feature < features; ++feature)
            if (!categorical[static_cast<size_t>(feature)])
                numerical_features_.push_back(feature);

        const int output_features = output_feature_count();
        std::vector<double> output(static_cast<size_t>(rows) * static_cast<size_t>(output_features), 0.0);
        copy_numerical_(X, rows, features, output);

        const int permutations = std::max(1, config.permutations);
        std::vector<int> order(static_cast<size_t>(rows));
        std::iota(order.begin(), order.end(), 0);
        std::mt19937_64 generator(config.seed);
        full_statistics_.resize(categorical_features_.size());

        for (size_t position = 0; position < categorical_features_.size(); ++position) {
            const int feature = categorical_features_[position];
            std::vector<double> encoded(static_cast<size_t>(rows), 0.0);
            for (int permutation = 0; permutation < permutations; ++permutation) {
                std::iota(order.begin(), order.end(), 0);
                std::shuffle(order.begin(), order.end(), generator);
                StatisticsMap prefix;
                prefix.reserve(static_cast<size_t>(rows / 2 + 1));
                for (int row : order) {
                    const uint64_t key = category_key_(
                        X[static_cast<size_t>(row) * static_cast<size_t>(features) + static_cast<size_t>(feature)]);
                    const auto found = prefix.find(key);
                    const Stat stat = found == prefix.end() ? Stat{} : found->second;
                    encoded[static_cast<size_t>(row)] += (stat.sum + config.prior * config.prior_weight) /
                                                         (static_cast<double>(stat.count) + config.prior_weight);
                    auto& update = prefix[key];
                    update.sum += target[row];
                    ++update.count;
                }
            }
            const int output_feature = static_cast<int>(numerical_features_.size() + position);
            for (int row = 0; row < rows; ++row) {
                output[static_cast<size_t>(row) * static_cast<size_t>(output_features) +
                       static_cast<size_t>(output_feature)] =
                    encoded[static_cast<size_t>(row)] / static_cast<double>(permutations);
            }

            auto& full = full_statistics_[position];
            full.reserve(static_cast<size_t>(rows / 2 + 1));
            for (int row = 0; row < rows; ++row) {
                const uint64_t key = category_key_(
                    X[static_cast<size_t>(row) * static_cast<size_t>(features) + static_cast<size_t>(feature)]);
                auto& stat = full[key];
                stat.sum += target[row];
                ++stat.count;
            }
        }
        prior_ = config.prior;
        prior_weight_ = config.prior_weight;
        fitted_ = true;
        return output;
    }

    std::vector<double> transform(const double* X, int rows, int features) const {
        if (!fitted_)
            throw std::runtime_error("OrderedCategoricalEncoder: encoder is not fitted");
        if (!X || rows < 0 || features != input_features_)
            throw std::invalid_argument("OrderedCategoricalEncoder: input shape mismatch");
        const int output_features = output_feature_count();
        std::vector<double> output(static_cast<size_t>(rows) * static_cast<size_t>(output_features), 0.0);
        copy_numerical_(X, rows, features, output);
        for (size_t position = 0; position < categorical_features_.size(); ++position) {
            const int feature = categorical_features_[position];
            const auto& statistics = full_statistics_[position];
            const int output_feature = static_cast<int>(numerical_features_.size() + position);
            for (int row = 0; row < rows; ++row) {
                const uint64_t key = category_key_(
                    X[static_cast<size_t>(row) * static_cast<size_t>(features) + static_cast<size_t>(feature)]);
                const auto found = statistics.find(key);
                const Stat stat = found == statistics.end() ? Stat{} : found->second;
                output[static_cast<size_t>(row) * static_cast<size_t>(output_features) +
                       static_cast<size_t>(output_feature)] =
                    (stat.sum + prior_ * prior_weight_) / (static_cast<double>(stat.count) + prior_weight_);
            }
        }
        return output;
    }

    [[nodiscard]] int output_feature_count() const noexcept {
        return static_cast<int>(numerical_features_.size() + categorical_features_.size());
    }
    [[nodiscard]] bool fitted() const noexcept {
        return fitted_;
    }

private:
    struct Stat {
        double sum = 0.0;
        int count = 0;
    };
    using StatisticsMap = std::unordered_map<uint64_t, Stat>;

    static uint64_t category_key_(double value) noexcept {
        if (std::isnan(value))
            return std::numeric_limits<uint64_t>::max();
        if (value == 0.0)
            value = 0.0;
        return std::bit_cast<uint64_t>(value);
    }

    static void validate_(const double* X, int rows, int features, const double* target,
                          const OrderedCategoricalConfig& config) {
        if (!X || !target || rows <= 0 || features <= 0)
            throw std::invalid_argument("OrderedCategoricalEncoder: invalid input");
        if (config.features.empty())
            throw std::invalid_argument("OrderedCategoricalEncoder: no categorical features configured");
        if (!(config.prior_weight > 0.0))
            throw std::invalid_argument("OrderedCategoricalEncoder: prior_weight must be positive");
    }

    void copy_numerical_(const double* X, int rows, int features, std::vector<double>& output) const {
        const int output_features = output_feature_count();
        for (int row = 0; row < rows; ++row)
            for (size_t position = 0; position < numerical_features_.size(); ++position)
                output[static_cast<size_t>(row) * static_cast<size_t>(output_features) + position] =
                    X[static_cast<size_t>(row) * static_cast<size_t>(features) +
                      static_cast<size_t>(numerical_features_[position])];
    }

    int input_features_ = 0;
    std::vector<int> categorical_features_;
    std::vector<int> numerical_features_;
    std::vector<StatisticsMap> full_statistics_;
    double prior_ = 0.0;
    double prior_weight_ = 1.0;
    bool fitted_ = false;
};

} // namespace foretree
