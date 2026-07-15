#include <atomic>
#include <cassert>
#include <cmath>
#include <vector>

#include "foretree/ensemble/forest.hpp"

namespace {

void test_compact_code_widths_match() {
    constexpr int rows = 32;
    constexpr int features = 2;
    std::vector<uint8_t> codes8(static_cast<size_t>(rows * features));
    std::vector<uint16_t> codes16(static_cast<size_t>(rows * features));
    std::vector<double> gradients(static_cast<size_t>(rows));
    std::vector<double> hessians(static_cast<size_t>(rows), 1.0);
    for (int row = 0; row < rows; ++row) {
        codes8[static_cast<size_t>(row * features)] = static_cast<uint8_t>(row % 8);
        codes8[static_cast<size_t>(row * features + 1)] = static_cast<uint8_t>((row * 3) % 8);
        codes16[static_cast<size_t>(row * features)] = codes8[static_cast<size_t>(row * features)];
        codes16[static_cast<size_t>(row * features + 1)] = codes8[static_cast<size_t>(row * features + 1)];
        gradients[static_cast<size_t>(row)] = row < rows / 2 ? -1.0 : 1.0;
    }
    auto narrow = foretree::QuantizedDataset::from_u8(rows, features, std::move(codes8), 8, {8, 8});
    auto wide = foretree::QuantizedDataset::from_u16(rows, features, std::move(codes16), 300, {300, 300});
    foretree::TreeConfig config;
    config.max_depth = 3;
    config.max_leaves = 4;
    config.min_samples_leaf = 2;
    config.n_bins = 300;
    foretree::UnifiedTree narrow_tree(config);
    foretree::UnifiedTree wide_tree(config);
    narrow_tree.fit(narrow, gradients, hessians);
    wide_tree.fit(wide, gradients, hessians);
    const auto narrow_prediction = narrow_tree.predict(narrow);
    const auto wide_prediction = wide_tree.predict(wide);
    assert(narrow_prediction.size() == wide_prediction.size());
    for (size_t i = 0; i < narrow_prediction.size(); ++i)
        assert(std::abs(narrow_prediction[i] - wide_prediction[i]) < 1.0e-12);
}

void test_validation_rows_need_not_match_training_rows() {
    constexpr int train_rows = 96;
    constexpr int valid_rows = 31;
    constexpr int features = 3;
    std::vector<double> train_x(static_cast<size_t>(train_rows * features));
    std::vector<double> train_y(static_cast<size_t>(train_rows));
    std::vector<double> valid_x(static_cast<size_t>(valid_rows * features));
    std::vector<double> valid_y(static_cast<size_t>(valid_rows));
    for (int row = 0; row < train_rows; ++row) {
        for (int feature = 0; feature < features; ++feature)
            train_x[static_cast<size_t>(row * features + feature)] = 0.01 * (row + feature);
        train_y[static_cast<size_t>(row)] = train_x[static_cast<size_t>(row * features)];
    }
    for (int row = 0; row < valid_rows; ++row) {
        for (int feature = 0; feature < features; ++feature)
            valid_x[static_cast<size_t>(row * features + feature)] = 0.02 * (row + feature);
        valid_y[static_cast<size_t>(row)] = valid_x[static_cast<size_t>(row * features)];
    }
    foretree::ForeForestConfig config;
    config.mode = foretree::ForeForestConfig::Mode::GBDT;
    config.objective = foretree::ForeForestConfig::Objective::SquaredError;
    config.n_estimators = 2;
    config.threads = 2;
    config.hist_cfg.max_bins = 16;
    config.tree_cfg.max_depth = 3;
    config.tree_cfg.max_leaves = 4;
    config.tree_cfg.min_samples_leaf = 2;
    config.early_stopping_enabled = true;
    config.early_stopping_rounds = 10;
    config.early_stopping_min_delta = 1.0e9;
    foretree::ForeForest forest(config);
    forest.fit_complete(train_x.data(), train_rows, features, train_y.data(), valid_x.data(), valid_rows, features,
                        valid_y.data());
    assert(forest.best_iteration() == 1);
    assert(forest.size() == 1);
    assert(!forest.early_stopped());
}

foretree::ForeForestConfig
make_config(int threads, foretree::ForeForestConfig::Device device = foretree::ForeForestConfig::Device::CPU) {
    foretree::ForeForestConfig config;
    config.mode = foretree::ForeForestConfig::Mode::GBDT;
    config.objective = foretree::ForeForestConfig::Objective::SquaredError;
    config.n_estimators = 8;
    config.learning_rate = 0.1;
    config.threads = threads;
    config.device = device;
    config.hist_cfg.method = "hist";
    config.hist_cfg.max_bins = 16;
    config.tree_cfg.max_depth = 4;
    config.tree_cfg.max_leaves = 8;
    config.tree_cfg.min_samples_split = 4;
    config.tree_cfg.min_samples_leaf = 2;
    config.tree_cfg.cache_threshold = 8;
    config.tree_cfg.cache_histograms = true;
    return config;
}

#ifdef FORETREE_HAS_CUDA
void test_cuda_training_matches_cpu() {
    if (!foretree::cuda::is_available())
        return;
    constexpr int rows = 256;
    constexpr int features = 6;
    std::vector<double> x(static_cast<size_t>(rows * features));
    std::vector<double> y(static_cast<size_t>(rows));
    for (int row = 0; row < rows; ++row) {
        for (int feature = 0; feature < features; ++feature) {
            x[static_cast<size_t>(row * features + feature)] =
                std::cos(0.017 * static_cast<double>((row + 3) * (feature + 1)));
        }
        y[static_cast<size_t>(row)] =
            x[static_cast<size_t>(row * features)] + 0.25 * x[static_cast<size_t>(row * features + 2)];
    }
    auto cpu_config = make_config(1);
    auto cuda_config = make_config(2, foretree::ForeForestConfig::Device::CUDA);
    cpu_config.n_estimators = cuda_config.n_estimators = 4;
    foretree::ForeForest cpu(cpu_config);
    foretree::ForeForest gpu(cuda_config);
    cpu.fit_complete(x.data(), rows, features, y.data());
    gpu.fit_complete(x.data(), rows, features, y.data());
    const auto cpu_prediction = cpu.predict(x.data(), rows, features);
    const auto gpu_prediction = gpu.predict(x.data(), rows, features);
    assert(cpu_prediction.size() == gpu_prediction.size());
    for (size_t i = 0; i < cpu_prediction.size(); ++i) {
        assert(std::abs(cpu_prediction[i] - gpu_prediction[i]) < 1.0e-5);
    }
}
#endif

void test_parallel_training_and_fused_inference_match_serial() {
    constexpr int rows = 512;
    constexpr int features = 8;
    std::vector<double> x(static_cast<size_t>(rows * features));
    std::vector<double> y(static_cast<size_t>(rows));
    for (int row = 0; row < rows; ++row) {
        for (int feature = 0; feature < features; ++feature) {
            x[static_cast<size_t>(row * features + feature)] =
                std::sin(0.013 * static_cast<double>((row + 1) * (feature + 2)));
        }
        y[static_cast<size_t>(row)] =
            2.0 * x[static_cast<size_t>(row * features)] - 0.5 * x[static_cast<size_t>(row * features + 3)];
    }

    foretree::ForeForest serial(make_config(1));
    foretree::ForeForest parallel(make_config(4));
    serial.fit_complete(x.data(), rows, features, y.data());
    parallel.fit_complete(x.data(), rows, features, y.data());

    const auto serial_prediction = serial.predict(x.data(), rows, features);
    const auto parallel_prediction = parallel.predict(x.data(), rows, features);
    assert(serial_prediction.size() == parallel_prediction.size());
    for (size_t i = 0; i < serial_prediction.size(); ++i) {
        assert(std::abs(serial_prediction[i] - parallel_prediction[i]) < 1e-12);
    }
}

void test_sibling_histogram_subtraction_matches_rebuild() {
    constexpr int rows = 192;
    constexpr int features = 5;
    std::vector<double> x(static_cast<size_t>(rows * features));
    std::vector<double> y(static_cast<size_t>(rows));
    for (int row = 0; row < rows; ++row) {
        for (int feature = 0; feature < features; ++feature)
            x[static_cast<size_t>(row * features + feature)] = std::sin(0.03 * (row + 1) * (feature + 1));
        y[static_cast<size_t>(row)] =
            x[static_cast<size_t>(row * features)] - 0.4 * x[static_cast<size_t>(row * features + 2)];
    }
    auto subtract_config = make_config(2);
    auto rebuild_config = subtract_config;
    subtract_config.n_estimators = rebuild_config.n_estimators = 3;
    subtract_config.tree_cfg.use_sibling_subtract = true;
    rebuild_config.tree_cfg.use_sibling_subtract = false;
    foretree::ForeForest subtract_model(subtract_config);
    foretree::ForeForest rebuild_model(rebuild_config);
    subtract_model.fit_complete(x.data(), rows, features, y.data());
    rebuild_model.fit_complete(x.data(), rows, features, y.data());
    const auto subtract_prediction = subtract_model.predict(x.data(), rows, features);
    const auto rebuild_prediction = rebuild_model.predict(x.data(), rows, features);
    for (size_t i = 0; i < subtract_prediction.size(); ++i)
        assert(std::abs(subtract_prediction[i] - rebuild_prediction[i]) < 1.0e-12);
}

void test_histogram_scratch_is_reused() {
    foretree::HistogramPool pool(64, 1, 2);
    auto first = pool.get();
    auto* address = first.get();
    pool.return_histogram(std::move(first));
    auto second = pool.get();
    assert(second.get() == address);
}

void test_nested_parallel_work_runs_without_deadlock() {
    auto executor = std::make_shared<foretree::ParallelExecutor>(4);
    std::atomic<int> visits = 0;
    executor->parallel_for(0, 4, 1, [&](int begin, int end) {
        for (int outer = begin; outer < end; ++outer) {
            (void)outer;
            executor->parallel_for(0, 8, 1, [&](int inner_begin, int inner_end) {
                visits.fetch_add(inner_end - inner_begin, std::memory_order_relaxed);
            });
        }
    });
    assert(visits.load(std::memory_order_relaxed) == 32);
}

} // namespace

int main() {
    test_compact_code_widths_match();
    test_validation_rows_need_not_match_training_rows();
    test_histogram_scratch_is_reused();
    test_nested_parallel_work_runs_without_deadlock();
    test_parallel_training_and_fused_inference_match_serial();
    test_sibling_histogram_subtraction_matches_rebuild();
#ifdef FORETREE_HAS_CUDA
    test_cuda_training_matches_cpu();
#endif
}
