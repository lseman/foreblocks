#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "foretree/core/parallel_executor.hpp"

namespace foretree {

struct HistogramOutputView {
    std::span<double> gradients;
    std::span<double> hessians;
    std::span<int> counts;
};

template <class Code, bool UnitHessian, bool WithCounts = true> struct FeatureMajorHistogramKernel {
    template <class RowAt>
    static void build(std::span<const Code> feature_major_codes, int dataset_rows, int row_count, RowAt&& row_at,
                      std::span<const int> active_features, std::span<const size_t> feature_offsets,
                      std::span<const int> missing_codes, std::span<const double> gradients,
                      std::span<const double> hessians, HistogramOutputView output, ParallelExecutor& executor) {
        const int feature_count = static_cast<int>(active_features.size());
        const int work = row_count * feature_count;
        const int grain = work >= 32768 ? 1 : std::max(1, feature_count);
        executor.parallel_for(0, feature_count, grain, [&](int feature_begin, int feature_end) {
            for (int position = feature_begin; position < feature_end; ++position) {
                const int feature = active_features[static_cast<size_t>(position)];
                const uint16_t missing = static_cast<uint16_t>(missing_codes[static_cast<size_t>(feature)]);
                const size_t histogram_offset = feature_offsets[static_cast<size_t>(feature)];
                const size_t column_offset = static_cast<size_t>(feature) * static_cast<size_t>(dataset_rows);

                for (int sample = 0; sample < row_count; ++sample) {
                    const int row = row_at(sample);
                    uint16_t bin = static_cast<uint16_t>(feature_major_codes[column_offset + static_cast<size_t>(row)]);
                    if (bin >= missing)
                        bin = missing;
                    const size_t bin_offset = histogram_offset + static_cast<size_t>(bin);
                    output.gradients[bin_offset] += gradients[static_cast<size_t>(row)];
                    if constexpr (WithCounts)
                        ++output.counts[bin_offset];
                    if constexpr (!UnitHessian)
                        output.hessians[bin_offset] += hessians[static_cast<size_t>(row)];
                }

                if constexpr (UnitHessian) {
                    const size_t end = histogram_offset + static_cast<size_t>(missing) + 1;
                    for (size_t bin = histogram_offset; bin < end; ++bin)
                        output.hessians[bin] = static_cast<double>(output.counts[bin]);
                }
            }
        });
    }
};

template <class Code, class RowAt>
void dispatch_feature_major_histogram(bool unit_hessian, std::span<const Code> feature_major_codes, int dataset_rows,
                                      int row_count, RowAt&& row_at, std::span<const int> active_features,
                                      std::span<const size_t> feature_offsets, std::span<const int> missing_codes,
                                      std::span<const double> gradients, std::span<const double> hessians,
                                      HistogramOutputView output, ParallelExecutor& executor) {
    if (unit_hessian) {
        FeatureMajorHistogramKernel<Code, true>::build(feature_major_codes, dataset_rows, row_count,
                                                       std::forward<RowAt>(row_at), active_features, feature_offsets,
                                                       missing_codes, gradients, hessians, output, executor);
    } else {
        FeatureMajorHistogramKernel<Code, false>::build(feature_major_codes, dataset_rows, row_count,
                                                        std::forward<RowAt>(row_at), active_features, feature_offsets,
                                                        missing_codes, gradients, hessians, output, executor);
    }
}

} // namespace foretree
