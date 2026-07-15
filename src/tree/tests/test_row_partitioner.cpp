#include <algorithm>
#include <cassert>
#include <vector>

#include "foretree/tree/row_partitioner.hpp"

int main() {
    std::vector<int> rows = {90, 1, 2, 3, 4, 5, 91};
    const int middle = foretree::RowPartitioner::partition(rows, 1, 6, [](int row) { return row % 2 == 0; });

    assert(middle == 3);
    assert(rows.front() == 90);
    assert(rows.back() == 91);
    assert(std::all_of(rows.begin() + 1, rows.begin() + middle, [](int row) { return row % 2 == 0; }));
    assert(std::none_of(rows.begin() + middle, rows.begin() + 6, [](int row) { return row % 2 == 0; }));
}
