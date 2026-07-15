#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

#include "foretree/split/split_finder.hpp"

int main() {
    // Two features, two finite bins plus one missing bin each. Feature 0 has
    // the strongest split, but feature bagging selects feature 1 only.
    const std::vector<double> gradients = {-10.0, 10.0, 0.0, -2.0, 2.0, 0.0};
    const std::vector<double> hessians = {10.0, 10.0, 0.0, 10.0, 10.0, 0.0};
    const std::vector<int> counts = {10, 10, 0, 10, 10, 0};
    const std::vector<size_t> offsets = {0, 3, 6};
    const std::vector<int> finite_bins = {2, 2};
    const std::vector<int> missing_bins = {2, 2};
    const std::vector<int> active_features = {1};

    foretree::splitx::SplitContext context;
    context.G = &gradients;
    context.H = &hessians;
    context.C = &counts;
    context.P = 2;
    context.Gp = 0.0;
    context.Hp = 20.0;
    context.Cp = 20;
    context.variable_bins = true;
    context.feature_offsets = offsets.data();
    context.finite_bins_per_feat = finite_bins.data();
    context.missing_ids_per_feat = missing_bins.data();
    context.active_features = &active_features;
    context.hyp.min_samples_leaf_ = 1;
    context.hyp.min_child_weight_ = 0.0;

    const foretree::AxisSplitFinder finder;
    const auto split = finder.best_axis(context);
    assert(split.feat == 1);
    assert(split.thr == 0);

    // Categorical subset search must honor semantic feature types. Numerical
    // histogram bins are ordered intervals and cannot be partitioned as an
    // arbitrary set.
    const std::vector<foretree::FeatureType> numerical_types = {foretree::FeatureType::Numerical,
                                                                foretree::FeatureType::Numerical};
    context.feature_types = numerical_types.data();
    const foretree::CategoricalPartitionSplitFinder categorical_finder;
    const auto rejected = categorical_finder.best_categorical_partition(context);
    assert(!std::isfinite(rejected.gain));

    const std::vector<foretree::FeatureType> mixed_types = {foretree::FeatureType::Numerical,
                                                            foretree::FeatureType::Categorical};
    context.feature_types = mixed_types.data();
    const auto categorical = categorical_finder.best_categorical_partition(context);
    assert(categorical.feat == 1);
    assert(categorical.kind == foretree::splitx::SplitKind::CategoricalPartition);
}
