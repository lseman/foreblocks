#pragma once

#include <algorithm>
#include <cmath>
#include <span>

namespace foretree::multiclass {

// The reference class has a fixed logit of zero and no stored margin.
inline double log_normalizer(std::span<const double> margins) {
    const double peak = std::max(0.0, *std::max_element(margins.begin(), margins.end()));
    double total = std::exp(-peak);
    for (double margin : margins)
        total += std::exp(margin - peak);
    return peak + std::log(total);
}

inline double logloss(std::span<const double> margins, int label) {
    const double target = label == static_cast<int>(margins.size()) ? 0.0 : margins[label];
    return log_normalizer(margins) - target;
}

inline void probabilities(std::span<const double> margins, std::span<double> output) {
    const double peak = std::max(0.0, *std::max_element(margins.begin(), margins.end()));
    double total = 0.0;
    for (size_t c = 0; c < margins.size(); ++c) {
        output[c] = std::exp(margins[c] - peak);
        total += output[c];
    }
    output.back() = std::exp(-peak);
    total += output.back();
    for (double& probability : output)
        probability /= total;
}

} // namespace foretree::multiclass
