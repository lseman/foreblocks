#include <cassert>
#include <cmath>
#include <vector>

#include "foretree/core/ordered_categorical.hpp"
#include "foretree/ensemble/forest.hpp"

int main() {
    constexpr int rows = 8;
    constexpr int features = 3;
    const std::vector<double> X = {
        0.0, 10.0, 1.0, 1.0, 11.0, 2.0, 0.0, 12.0, 3.0, 1.0, 13.0, 4.0,
        0.0, 14.0, 5.0, 1.0, 15.0, 6.0, 0.0, 16.0, 7.0, 1.0, 17.0, 8.0,
    };
    const std::vector<double> y = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};

    foretree::OrderedCategoricalConfig encoder_config;
    encoder_config.enabled = true;
    encoder_config.features = {0};
    encoder_config.permutations = 2;
    encoder_config.prior = 0.5;
    encoder_config.prior_weight = 1.0;
    foretree::OrderedCategoricalEncoder encoder;
    const auto train = encoder.fit_transform(X.data(), rows, features, y.data(), encoder_config);
    assert(encoder.output_feature_count() == features);
    assert(train.size() == X.size());
    for (double value : train)
        assert(std::isfinite(value));

    const auto inference = encoder.transform(X.data(), rows, features);
    // Both categories have a full-data target mean of 0 or 1 respectively,
    // smoothed toward the 0.5 prior.
    assert(std::abs(inference[2] - 0.1) < 1e-12);
    assert(std::abs(inference[5] - 0.9) < 1e-12);

    foretree::ForeForestConfig forest_config;
    forest_config.mode = foretree::ForeForestConfig::Mode::GBDT;
    forest_config.n_estimators = 4;
    forest_config.dart_enabled = false;
    forest_config.ordered_categorical_enabled = true;
    forest_config.categorical_features = {0};
    forest_config.ordered_categorical_permutations = 2;
    forest_config.ordered_categorical_prior = 0.5;
    forest_config.ordered_boosting_enabled = true;
    forest_config.ordered_boosting_min_prefix = 0.5;
    forest_config.tree_cfg.max_depth = 2;
    forest_config.tree_cfg.max_leaves = 4;
    forest_config.tree_cfg.min_samples_split = 2;
    forest_config.tree_cfg.min_samples_leaf = 1;
    forest_config.hist_cfg.max_bins = 8;
    foretree::ForeForest forest(forest_config);
    forest.fit_complete(X.data(), rows, features, y.data());
    const auto prediction = forest.predict(X.data(), rows, features);
    assert(prediction.size() == static_cast<size_t>(rows));
    for (double value : prediction)
        assert(std::isfinite(value));

    constexpr int sparse_rows = 64;
    constexpr int sparse_features = 5;
    std::vector<double> sparse_X(static_cast<size_t>(sparse_rows * sparse_features), 0.0);
    std::vector<double> sparse_y(static_cast<size_t>(sparse_rows));
    for (int row = 0; row < sparse_rows; ++row) {
        sparse_X[static_cast<size_t>(row * sparse_features)] = row % 3;
        const int active = 1 + row % 4;
        sparse_X[static_cast<size_t>(row * sparse_features + active)] = 1.0 + 0.01 * row;
        sparse_y[static_cast<size_t>(row)] = static_cast<double>(row % 3) + 0.2 * active;
    }
    forest_config.efb_enabled = true;
    forest_config.efb_sparse_threshold = 0.3;
    forest_config.efb_min_nonzero = 4;
    forest_config.n_estimators = 3;
    foretree::ForeForest combined(forest_config);
    combined.fit_complete(sparse_X.data(), sparse_rows, sparse_features, sparse_y.data());
    const auto combined_prediction = combined.predict(sparse_X.data(), sparse_rows, sparse_features);
    assert(combined_prediction.size() == static_cast<size_t>(sparse_rows));
    for (double value : combined_prediction)
        assert(std::isfinite(value));
}
