#include <cassert>
#include <cmath>
#include <vector>

#include "foretree/gpu/cuda_histogram.hpp"

int main() {
    if (!foretree::cuda::is_available())
        return 0;

    auto dataset = foretree::QuantizedDataset::from_u8(4, 2, {0, 1, 0, 0, 1, 1, 2, 0}, 2, {2, 2});
    foretree::cuda::CudaHistogramEngine engine(dataset);
    engine.set_gradients(std::vector<float>{-2, -1, 1, 2}, std::vector<float>{1, 1, 1, 1});

    const auto histogram = engine.build_histogram();
    assert(histogram.gradients.size() == 6);
    assert(std::abs(histogram.gradients[0] + 3.0F) < 1.0e-6F);
    assert(std::abs(histogram.gradients[1] - 1.0F) < 1.0e-6F);
    assert(std::abs(histogram.gradients[2] - 2.0F) < 1.0e-6F);
    assert(histogram.counts[0] == 2);
    assert(histogram.counts[1] == 1);
    assert(histogram.counts[2] == 1);

    const auto split = engine.find_best_split(0.0F, 4.0F, 1.0F, 1, 0.0F);
    assert(split.gain >= 0.0F);
    const auto routes = engine.route_rows(split);
    assert(routes.size() == 4);

    engine.set_gradients(std::vector<float>{-2, -1, 1, 2}, std::vector<float>{1, 1, 1, 1});
    const std::vector<foretree::cuda::FeaturePair> pairs = {{0, 1}};
    const auto joint = engine.build_joint_histograms(pairs, 2);
    assert(joint.pair_count == 1);
    assert(joint.reduced_bins == 2);
    assert(joint.stride() == 5);
    assert(joint.counts == std::vector<uint32_t>({1, 1, 0, 1, 1}));
    assert(std::abs(joint.gradients[0] + 1.0F) < 1.0e-6F);
    assert(std::abs(joint.gradients[1] + 2.0F) < 1.0e-6F);
    assert(std::abs(joint.gradients[3] - 1.0F) < 1.0e-6F);
    assert(std::abs(joint.gradients[4] - 2.0F) < 1.0e-6F);

    const auto subset_joint = engine.build_joint_histograms(pairs, 2, std::vector<uint32_t>{0, 2});
    assert(subset_joint.counts == std::vector<uint32_t>({0, 1, 0, 1, 0}));

    std::vector<float> labels = {0, 1, 0, 1};
    std::vector<float> margins(4, 0.0F);
    engine.compute_binary_logloss_gradients(labels, margins);
    const auto binary_histogram = engine.build_histogram();
    assert(std::isfinite(binary_histogram.gradients[0]));
    assert(std::isfinite(binary_histogram.hessians[0]));
}
