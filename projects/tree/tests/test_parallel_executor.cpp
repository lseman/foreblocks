// ParallelExecutor: every index runs exactly once, back-to-back jobs, nested
// and concurrent callers, exceptions, and construct/destroy races.
#include <atomic>
#include <cassert>
#include <stdexcept>
#include <thread>
#include <vector>

#include "foretree/core/parallel_executor.hpp"

namespace {

void test_each_index_once(foretree::ParallelExecutor& ex) {
    for (int round = 0; round < 2000; ++round) {
        const int n = 1 + round % 97;
        std::vector<std::atomic<int>> hits(static_cast<size_t>(n));
        ex.parallel_for(0, n, 1 + round % 5, [&](int b, int e) {
            for (int i = b; i < e; ++i) hits[static_cast<size_t>(i)].fetch_add(1);
        });
        for (auto& h : hits) assert(h.load() == 1);
    }
}

void test_nested_calls_run_inline(foretree::ParallelExecutor& ex) {
    std::atomic<long> total{0};
    ex.parallel_for(0, 64, 1, [&](int b, int e) {
        for (int i = b; i < e; ++i)
            ex.parallel_for(0, 100, 1, [&](int ib, int ie) { total += ie - ib; });
    });
    assert(total.load() == 64 * 100);
}

void test_concurrent_callers(foretree::ParallelExecutor& ex) {
    std::atomic<long> total{0};
    std::vector<std::thread> callers;
    for (int t = 0; t < 4; ++t)
        callers.emplace_back([&] {
            for (int r = 0; r < 300; ++r)
                ex.parallel_for(0, 1000, 10, [&](int b, int e) { total += e - b; });
        });
    for (auto& c : callers) c.join();
    assert(total.load() == 4L * 300 * 1000);
}

void test_exception_propagates(foretree::ParallelExecutor& ex) {
    bool caught = false;
    try {
        ex.parallel_for(0, 100, 1, [](int b, int e) {
            if (b <= 42 && 42 < e) throw std::runtime_error("chunk failed");
        });
    } catch (const std::runtime_error&) {
        caught = true;
    }
    assert(caught);
    std::atomic<int> after{0};
    ex.parallel_for(0, 50, 1, [&](int b, int e) { after += e - b; });  // still usable
    assert(after.load() == 50);
}

}  // namespace

int main() {
    for (int i = 0; i < 200; ++i) {  // construct/destroy races
        foretree::ParallelExecutor ex(1 + i % 8);
        if (i % 2) ex.parallel_for(0, 16, 1, [](int, int) {});
    }
    foretree::ParallelExecutor ex(8);
    test_each_index_once(ex);
    test_nested_calls_run_inline(ex);
    test_concurrent_callers(ex);
    test_exception_propagates(ex);
    return 0;
}
