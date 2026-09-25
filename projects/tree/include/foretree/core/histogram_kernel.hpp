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

// Per-thread scratch that only grows: large std::vector allocations are
// mmap-backed, so allocating (and zero-filling) them per histogram costs page
// faults on every node.
template <class T> inline T* histogram_scratch(size_t count, int slot) {
    static thread_local std::vector<T> buffers[4];
    auto& buffer = buffers[slot];
    if (buffer.size() < count)
        buffer.resize(count);
    return buffer.data();
}

template <class Code, bool UnitHessian, bool WithCounts = true> struct FeatureMajorHistogramKernel {
    template <class RowAt>
    static void build(std::span<const Code> feature_major_codes, int dataset_rows, int row_count, RowAt&& row_at,
                      std::span<const int> active_features, std::span<const size_t> feature_offsets,
                      std::span<const int> missing_codes, std::span<const double> gradients,
                      std::span<const double> hessians, HistogramOutputView output, ParallelExecutor& executor) {
        const int feature_count = static_cast<int>(active_features.size());
        const int work = row_count * feature_count;
        const int grain = work >= 2048 ? 1 : std::max(1, feature_count);
        // Gather the node's rows and gradients into contiguous order once
        // ("ordered gradients"), so each feature pass reads them sequentially
        // instead of re-gathering g[row] / h[row] through the row index.
        // (Measured: reading g[row] directly for small nodes is slower end to
        // end; mid-size nodes dominate and prefer the ordered buffer.)
        int* rows = histogram_scratch<int>(static_cast<size_t>(row_count), 0);
        double* ordered_g = histogram_scratch<double>(static_cast<size_t>(row_count), 0);
        double* ordered_h = UnitHessian ? nullptr : histogram_scratch<double>(static_cast<size_t>(row_count), 1);
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
        const size_t part_total = part_size * static_cast<size_t>(row_blocks);
        double* part_g = histogram_scratch<double>(part_total, 2);
        double* part_h = UnitHessian ? nullptr : histogram_scratch<double>(part_total, 3);
        int* part_c = WithCounts ? histogram_scratch<int>(part_total, 1) : nullptr;
        executor.parallel_for(0, feature_count * row_blocks, 1, [&](int task_begin, int task_end) {
            for (int task = task_begin; task < task_end; ++task) {
                const int position = task / row_blocks;
                const int block = task % row_blocks;
                const int begin = static_cast<int>(static_cast<int64_t>(row_count) * block / row_blocks);
                const int end = static_cast<int>(static_cast<int64_t>(row_count) * (block + 1) / row_blocks);
                const size_t base = static_cast<size_t>(block) * part_size + part_offsets[static_cast<size_t>(position)];
                const size_t bins = part_offsets[static_cast<size_t>(position) + 1] -
                                    part_offsets[static_cast<size_t>(position)];
                std::fill(part_g + base, part_g + base + bins, 0.0);
                if constexpr (!UnitHessian)
                    std::fill(part_h + base, part_h + base + bins, 0.0);
                if constexpr (WithCounts)
                    std::fill(part_c + base, part_c + base + bins, 0);
                accumulate(position, begin, end, part_g + base, UnitHessian ? nullptr : part_h + base,
                           WithCounts ? part_c + base : nullptr);
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

// Quantized gradients (LightGBM-style quantized training): per row, g and h
// are small integers packed in one int32 (g signed in the high 16 bits, h
// unsigned in the low 16). One int64 add per row updates both bin sums (g in
// the high 32 bits, h in the low 32); sums are rescaled to doubles per
// feature. Gathering 4 bytes per row instead of 16 and one integer add
// instead of two double adds is ~1.1-1.8x faster than the double kernel.
// Overflow-safe while a bin holds < kQuantizedMaxRows rows of 8-bit levels.
struct QuantizedGradients {
    const int32_t* packed = nullptr;  // per dataset row
    double g_scale = 1.0;              // gradient = level * g_scale
    double h_scale = 1.0;              // hessian = level * h_scale
};
inline constexpr int kQuantizedMaxRows = 16'000'000;

inline int32_t pack_quantized(int g_level, int h_level) noexcept {
    return static_cast<int32_t>((static_cast<uint32_t>(g_level) << 16) | (static_cast<uint32_t>(h_level) & 0xFFFFU));
}

template <class Code, bool WithCounts = true> struct QuantizedFeatureMajorHistogramKernel {
    template <class RowAt>
    static void build(std::span<const Code> feature_major_codes, int dataset_rows, int row_count, RowAt&& row_at,
                      std::span<const int> active_features, std::span<const size_t> feature_offsets,
                      std::span<const int> missing_codes, const QuantizedGradients& quantized,
                      HistogramOutputView output, ParallelExecutor& executor) {
        const int feature_count = static_cast<int>(active_features.size());
        if (row_count <= 0 || feature_count == 0)
            return;
        const int64_t work = static_cast<int64_t>(row_count) * feature_count;
        const int grain = work >= 2048 ? 1 : std::max(1, feature_count);

        int* rows = histogram_scratch<int>(static_cast<size_t>(row_count), 2);
        int32_t* ordered = histogram_scratch<int32_t>(static_cast<size_t>(row_count), 0);
        const int gather_grain = std::max(4096, row_count / std::max(1, static_cast<int>(executor.thread_count())));
        executor.parallel_for(0, row_count, gather_grain, [&](int begin, int end) {
            for (int sample = begin; sample < end; ++sample) {
                const int row = row_at(sample);
                rows[static_cast<size_t>(sample)] = row;
                ordered[static_cast<size_t>(sample)] = quantized.packed[static_cast<size_t>(row)];
            }
        });

        // Integer bin sums of samples [begin, end) for one feature; `emit(bin,
        // g_sum, h_sum, count)` receives each bin.
        auto accumulate = [&](int position, int begin, int end, auto&& emit) {
            const int feature = active_features[static_cast<size_t>(position)];
            const int missing = missing_codes[static_cast<size_t>(feature)];
            static thread_local std::vector<int64_t> acc;
            static thread_local std::vector<int32_t> cnt;
            acc.assign(static_cast<size_t>(missing) + 1, 0);
            cnt.assign(static_cast<size_t>(missing) + 1, 0);
            int64_t* a = acc.data();
            int32_t* c = cnt.data();
            const Code* column = feature_major_codes.data() + static_cast<size_t>(feature) * dataset_rows;
            for (int sample = begin; sample < end; ++sample) {
                const int code = static_cast<int>(column[rows[static_cast<size_t>(sample)]]);
                const int bin = code >= missing ? missing : code;
                const int32_t v = ordered[static_cast<size_t>(sample)];
                a[bin] += (static_cast<int64_t>(v >> 16) << 32) + static_cast<int64_t>(static_cast<uint16_t>(v));
                ++c[bin];
            }
            for (int bin = 0; bin <= missing; ++bin) {
                const int64_t h_sum = a[bin] & 0xFFFFFFFFLL;
                const int64_t g_sum = (a[bin] - h_sum) >> 32;
                emit(bin, g_sum, h_sum, c[bin]);
            }
        };

        const int threads = std::max(1, static_cast<int>(executor.thread_count()));
        constexpr int kMinRowsPerBlock = 16384;
        const int row_blocks =
            std::clamp((2 * threads + feature_count - 1) / feature_count, 1,
                       std::max(1, row_count / kMinRowsPerBlock));

        if (row_blocks == 1) {
            executor.parallel_for(0, feature_count, grain, [&](int p0, int p1) {
                for (int p = p0; p < p1; ++p) {
                    const size_t offset = feature_offsets[static_cast<size_t>(active_features[static_cast<size_t>(p)])];
                    accumulate(p, 0, row_count, [&](int bin, int64_t g, int64_t h, int32_t c) {
                        output.gradients[offset + bin] += static_cast<double>(g) * quantized.g_scale;
                        output.hessians[offset + bin] += static_cast<double>(h) * quantized.h_scale;
                        if constexpr (WithCounts)
                            output.counts[offset + bin] += c;
                    });
                }
            });
            return;
        }

        // Few features, many rows: (feature, row block) tasks write integer
        // partials (exact as doubles), summed in block order then rescaled.
        std::vector<size_t> part_offsets(static_cast<size_t>(feature_count) + 1, 0);
        for (int p = 0; p < feature_count; ++p)
            part_offsets[static_cast<size_t>(p) + 1] =
                part_offsets[static_cast<size_t>(p)] +
                static_cast<size_t>(missing_codes[static_cast<size_t>(active_features[static_cast<size_t>(p)])]) + 1;
        const size_t part_size = part_offsets.back();
        const size_t part_total = part_size * static_cast<size_t>(row_blocks);
        double* part_g = histogram_scratch<double>(part_total, 2);
        double* part_h = histogram_scratch<double>(part_total, 3);
        int* part_c = histogram_scratch<int>(part_total, 1);
        executor.parallel_for(0, feature_count * row_blocks, 1, [&](int t0, int t1) {
            for (int task = t0; task < t1; ++task) {
                const int p = task / row_blocks, block = task % row_blocks;
                const int begin = static_cast<int>(static_cast<int64_t>(row_count) * block / row_blocks);
                const int end = static_cast<int>(static_cast<int64_t>(row_count) * (block + 1) / row_blocks);
                const size_t base = static_cast<size_t>(block) * part_size + part_offsets[static_cast<size_t>(p)];
                accumulate(p, begin, end, [&](int bin, int64_t g, int64_t h, int32_t c) {
                    part_g[base + bin] = static_cast<double>(g);
                    part_h[base + bin] = static_cast<double>(h);
                    part_c[base + bin] = c;
                });
            }
        });
        executor.parallel_for(0, feature_count, 1, [&](int p0, int p1) {
            for (int p = p0; p < p1; ++p) {
                const size_t offset = feature_offsets[static_cast<size_t>(active_features[static_cast<size_t>(p)])];
                const size_t bins = part_offsets[static_cast<size_t>(p) + 1] - part_offsets[static_cast<size_t>(p)];
                for (size_t bin = 0; bin < bins; ++bin) {
                    double g = 0.0, h = 0.0;
                    int c = 0;
                    for (int block = 0; block < row_blocks; ++block) {
                        const size_t at = static_cast<size_t>(block) * part_size + part_offsets[static_cast<size_t>(p)] + bin;
                        g += part_g[at];
                        h += part_h[at];
                        c += part_c[at];
                    }
                    output.gradients[offset + bin] += g * quantized.g_scale;
                    output.hessians[offset + bin] += h * quantized.h_scale;
                    if constexpr (WithCounts)
                        output.counts[offset + bin] += c;
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
