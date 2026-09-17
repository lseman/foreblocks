#pragma once

#include <utility>
#include <vector>

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
};

} // namespace foretree
