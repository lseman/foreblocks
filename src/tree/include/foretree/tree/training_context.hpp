#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "foretree/core/dataset.hpp"
#include "foretree/core/parallel_executor.hpp"

namespace foretree {

struct ExactSplitScratch {
    std::vector<double> missing_gradients;
    std::vector<double> missing_hessians;
    std::vector<int> missing_counts;
    std::vector<int> rows;
    std::vector<double> local_matrix;
    std::vector<double> local_gradients;
    std::vector<double> local_hessians;
    std::vector<uint8_t> local_missing;
    std::vector<double> column_storage;
    std::vector<const double*> columns;
    std::vector<int> local_indices;
    std::vector<double> local_missing_gradients;
    std::vector<double> local_missing_hessians;
    std::vector<int> local_missing_counts;

    void clear_logical_state() {
        rows.clear();
        local_missing.clear();
    }
};

// Tree-scoped reusable storage. Capacities survive across node evaluations and
// are released together when training of the tree is finished.
struct TreeTrainingArena {
    std::vector<int> histogram_rows;
    ExactSplitScratch exact;

    void reset() {
        histogram_rows.clear();
        exact.clear_logical_state();
    }
};

struct TreeTrainingContext {
    const QuantizedDataset* dataset = nullptr;
    std::span<const double> gradients;
    std::span<const double> hessians;
    std::shared_ptr<ParallelExecutor> executor;
    std::shared_ptr<TreeTrainingArena> arena = std::make_shared<TreeTrainingArena>();
    int rows = 0;
    int features = 0;
    int outputs = 1;
    bool unit_hessian = false;

    void bind(const QuantizedDataset& data, std::span<const double> gradient, std::span<const double> hessian,
              std::shared_ptr<ParallelExecutor> parallel_executor, int output_count) {
        dataset = &data;
        gradients = gradient;
        hessians = hessian;
        executor = std::move(parallel_executor);
        rows = data.rows();
        features = data.features();
        outputs = output_count;
        unit_hessian = std::all_of(hessians.begin(), hessians.end(), [](double value) { return value == 1.0; });
        arena->reset();
    }

    void release_dataset() noexcept {
        dataset = nullptr;
        gradients = {};
        hessians = {};
    }
};

} // namespace foretree
