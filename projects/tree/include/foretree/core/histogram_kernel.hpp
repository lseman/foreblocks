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
        // Gather the node's rows and gradients into contiguous order once
        // ("ordered gradients"), so each feature pass reads them sequentially
        // instead of re-gathering g[row] / h[row] through the row index.
        std::vector<int> rows(static_cast<size_t>(row_count));
        std::vector<double> ordered_g(static_cast<size_t>(row_count));
        std::vector<double> ordered_h(UnitHessian ? 0 : static_cast<size_t>(row_count));
        const int gather_grain = std::max(4096, row_count / std::max(1, static_cast<int>(executor.thread_count())));
        executor.parallel_for(0, row_count, gather_grain, [&](int begin, int end) {
            for (int sample = begin; sample < end; ++sample) {
                const int row = row_at(sample);
                rows[static_cast<size_t>(sample)] = row;
                ordered_g[static_cast<size_t>(sample)] = gradients[static_cast<size_t>(row)];
                if constexpr (!UnitHessian)
                    ordered_h[static_cast<size_t>(sample)] = hessians[static_cast<size_t>(row)];
            }
        });
        // Accumulate samples [begin, end) of one feature into (g, h, c), which
        // point at that feature's first bin.
        auto accumulate = [&](int position, int begin, int end, double* out_g, double* out_h, int* out_c) {
            const int feature = active_features[static_cast<size_t>(position)];
            const uint16_t missing = static_cast<uint16_t>(missing_codes[static_cast<size_t>(feature)]);
            const Code* column = feature_major_codes.data() + static_cast<size_t>(feature) * dataset_rows;
            for (int sample = begin; sample < end; ++sample) {
                const uint16_t code = static_cast<uint16_t>(column[rows[static_cast<size_t>(sample)]]);
                const size_t bin = code >= missing ? missing : code;
                out_g[bin] += ordered_g[static_cast<size_t>(sample)];
                if constexpr (WithCounts)
                    ++out_c[bin];
                if constexpr (!UnitHessian)
                    out_h[bin] += ordered_h[static_cast<size_t>(sample)];
            }
        };
        auto finish_feature = [&](int position) {
            if constexpr (UnitHessian) {
                const int feature = active_features[static_cast<size_t>(position)];
                const size_t begin = feature_offsets[static_cast<size_t>(feature)];
                const size_t end = begin + static_cast<size_t>(missing_codes[static_cast<size_t>(feature)]) + 1;
                for (size_t bin = begin; bin < end; ++bin)
                    output.hessians[bin] = static_cast<double>(output.counts[bin]);
            }
        };

        // Few features but many rows: splitting only by feature leaves threads
        // idle, so large nodes are also split into row blocks. Each (feature,
        // block) task fills a private partial histogram; partials are summed.
        const int threads = std::max(1, static_cast<int>(executor.thread_count()));
        constexpr int kMinRowsPerBlock = 16384;
        const int row_blocks =
            feature_count == 0
                ? 1
                : std::clamp((2 * threads + feature_count - 1) / feature_count, 1,
                             std::max(1, row_count / kMinRowsPerBlock));

        if (row_blocks == 1) {
            executor.parallel_for(0, feature_count, grain, [&](int feature_begin, int feature_end) {
                for (int position = feature_begin; position < feature_end; ++position) {
                    const int feature = active_features[static_cast<size_t>(position)];
                    const size_t offset = feature_offsets[static_cast<size_t>(feature)];
                    accumulate(position, 0, row_count, output.gradients.data() + offset,
                               output.hessians.data() + offset,
                               WithCounts ? output.counts.data() + offset : nullptr);
                    finish_feature(position);
                }
            });
            return;
        }

        // Partial layout: [block][active feature][bin], bins = missing code + 1.
        std::vector<size_t> part_offsets(static_cast<size_t>(feature_count) + 1, 0);
        for (int position = 0; position < feature_count; ++position) {
            const int feature = active_features[static_cast<size_t>(position)];
            part_offsets[static_cast<size_t>(position) + 1] =
                part_offsets[static_cast<size_t>(position)] +
                static_cast<size_t>(missing_codes[static_cast<size_t>(feature)]) + 1;
        }
        const size_t part_size = part_offsets.back();
        std::vector<double> part_g(part_size * static_cast<size_t>(row_blocks), 0.0);
        std::vector<double> part_h(UnitHessian ? 0 : part_size * static_cast<size_t>(row_blocks), 0.0);
        std::vector<int> part_c(WithCounts ? part_size * static_cast<size_t>(row_blocks) : 0, 0);
        executor.parallel_for(0, feature_count * row_blocks, 1, [&](int task_begin, int task_end) {
            for (int task = task_begin; task < task_end; ++task) {
                const int position = task / row_blocks;
                const int block = task % row_blocks;
                const int begin = static_cast<int>(static_cast<int64_t>(row_count) * block / row_blocks);
                const int end = static_cast<int>(static_cast<int64_t>(row_count) * (block + 1) / row_blocks);
                const size_t base = static_cast<size_t>(block) * part_size + part_offsets[static_cast<size_t>(position)];
                accumulate(position, begin, end, part_g.data() + base,
                           UnitHessian ? nullptr : part_h.data() + base,
                           WithCounts ? part_c.data() + base : nullptr);
            }
        });
        executor.parallel_for(0, feature_count, 1, [&](int feature_begin, int feature_end) {
            for (int position = feature_begin; position < feature_end; ++position) {
                const int feature = active_features[static_cast<size_t>(position)];
                const size_t offset = feature_offsets[static_cast<size_t>(feature)];
                const size_t bins = part_offsets[static_cast<size_t>(position) + 1] -
                                    part_offsets[static_cast<size_t>(position)];
                for (int block = 0; block < row_blocks; ++block) {
                    const size_t base =
                        static_cast<size_t>(block) * part_size + part_offsets[static_cast<size_t>(position)];
                    for (size_t bin = 0; bin < bins; ++bin) {
                        output.gradients[offset + bin] += part_g[base + bin];
                        if constexpr (!UnitHessian)
                            output.hessians[offset + bin] += part_h[base + bin];
                        if constexpr (WithCounts)
                            output.counts[offset + bin] += part_c[base + bin];
                    }
                }
                finish_feature(position);
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
