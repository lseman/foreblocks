#pragma once

#include <algorithm>
#include <utility>
#include <vector>

#include "foretree/core/parallel_executor.hpp"

namespace foretree {

// In-place unstable partition over a node's contiguous slice in the shared
// row-index arena. Keeping this primitive independent of split semantics makes
// every split kind use the same well-tested movement algorithm.
class RowPartitioner {
public:
    template <class Predicate> static int partition(std::vector<int>& rows, int begin, int end, Predicate&& goes_left) {
        int left = begin;
        int right = end - 1;
        while (left <= right) {
            if (goes_left(rows[static_cast<size_t>(left)])) {
                ++left;
                continue;
            }
            if (!goes_left(rows[static_cast<size_t>(right)])) {
                --right;
                continue;
            }
            std::swap(rows[static_cast<size_t>(left++)], rows[static_cast<size_t>(right--)]);
        }
        return left;
    }

    // Stable partition of rows[begin, end): rows going left keep their order,
    // then rows going right keep theirs. Keeping node slices in ascending row
    // order makes later per-feature histogram gathers nearly sequential.
    // Large slices are partitioned in parallel (per-block counts, then a
    // scatter into `scratch`); `goes_left` must be safe to call concurrently.
    template <class Predicate>
    static int stable_partition(std::vector<int>& rows, std::vector<int>& scratch, int begin, int end,
                                Predicate&& goes_left, ParallelExecutor* executor = nullptr) {
        const int n = end - begin;
        if (n <= 0)
            return begin;
        if (scratch.size() < static_cast<size_t>(n))
            scratch.resize(static_cast<size_t>(n));
        int* data = rows.data() + begin;
        int* tmp = scratch.data();

        constexpr int kParallelMinRows = 32768;
        constexpr int kMinBlockRows = 8192;
        const int threads = executor ? static_cast<int>(executor->thread_count()) : 1;
        const int blocks = std::clamp(n / kMinBlockRows, 1, std::max(1, 2 * threads));
        if (n < kParallelMinRows || blocks == 1 || threads <= 1) {
            int write = 0, right = 0;
            for (int i = 0; i < n; ++i) {
                const int row = data[i];
                if (goes_left(row))
                    data[write++] = row;  // write <= i: safe in place
                else
                    tmp[right++] = row;
            }
            std::copy(tmp, tmp + right, data + write);
            return begin + write;
        }

        std::vector<int> left_counts(static_cast<size_t>(blocks), 0);
        std::vector<unsigned char> side(static_cast<size_t>(n));
        auto block_range = [&](int b) {
            return std::pair<int, int>{static_cast<int>(static_cast<int64_t>(n) * b / blocks),
                                       static_cast<int>(static_cast<int64_t>(n) * (b + 1) / blocks)};
        };
        executor->parallel_for(0, blocks, 1, [&](int b0, int b1) {
            for (int b = b0; b < b1; ++b) {
                const auto [lo, hi] = block_range(b);
                int count = 0;
                for (int i = lo; i < hi; ++i) {
                    const bool left = goes_left(data[i]);
                    side[static_cast<size_t>(i)] = static_cast<unsigned char>(left);
                    count += left;
                }
                left_counts[static_cast<size_t>(b)] = count;
            }
        });
        std::vector<int> left_offsets(static_cast<size_t>(blocks) + 1, 0);
        for (int b = 0; b < blocks; ++b)
            left_offsets[static_cast<size_t>(b) + 1] =
                left_offsets[static_cast<size_t>(b)] + left_counts[static_cast<size_t>(b)];
        const int total_left = left_offsets.back();
        executor->parallel_for(0, blocks, 1, [&](int b0, int b1) {
            for (int b = b0; b < b1; ++b) {
                const auto [lo, hi] = block_range(b);
                int l = left_offsets[static_cast<size_t>(b)];
                int r = total_left + (lo - left_offsets[static_cast<size_t>(b)]);
                for (int i = lo; i < hi; ++i) {
                    if (side[static_cast<size_t>(i)])
                        tmp[l++] = data[i];
                    else
                        tmp[r++] = data[i];
                }
            }
        });
        executor->parallel_for(0, blocks, 1, [&](int b0, int b1) {
            const auto [lo, unused_hi] = block_range(b0);
            const auto [unused_lo, hi] = block_range(b1 - 1);
            (void)unused_hi;
            (void)unused_lo;
            std::copy(tmp + lo, tmp + hi, data + lo);
        });
        return begin + total_left;
    }
};

} // namespace foretree
