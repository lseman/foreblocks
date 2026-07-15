#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

namespace foretree {

class ParallelExecutor {
   public:
    explicit ParallelExecutor(
        unsigned thread_count = std::thread::hardware_concurrency()) {
        const unsigned count = std::max(1U, thread_count);
        workers_.reserve(count);
        for (unsigned i = 0; i < count; ++i) {
            workers_.emplace_back([this] { worker_loop_(); });
        }
    }

    ParallelExecutor(const ParallelExecutor&) = delete;
    ParallelExecutor& operator=(const ParallelExecutor&) = delete;

    ~ParallelExecutor() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

    [[nodiscard]] unsigned thread_count() const noexcept {
        return static_cast<unsigned>(workers_.size());
    }

    template <class Function>
    void parallel_for(int begin, int end, int minimum_grain,
                      Function&& function) {
        const int count = end - begin;
        if (count <= 0) return;

        const int grain = std::max(1, minimum_grain);
        if (count <= grain || workers_.size() <= 1 ||
            active_executor_ == this) {
            function(begin, end);
            return;
        }

        const int max_chunks = std::min<int>(static_cast<int>(workers_.size()),
                                             (count + grain - 1) / grain);
        const int chunk_size = (count + max_chunks - 1) / max_chunks;
        std::vector<std::future<void>> completions;
        completions.reserve(static_cast<size_t>(max_chunks));

        for (int chunk_begin = begin; chunk_begin < end;
             chunk_begin += chunk_size) {
            const int chunk_end = std::min(end, chunk_begin + chunk_size);
            completions.push_back(enqueue_([&, chunk_begin, chunk_end] {
                function(chunk_begin, chunk_end);
            }));
        }
        for (auto& completion : completions) completion.get();
    }

   private:
    template <class Function>
    std::future<void> enqueue_(Function&& function) {
        auto task = std::make_shared<std::packaged_task<void()>>(
            std::forward<Function>(function));
        auto completion = task->get_future();
        {
            std::lock_guard lock(mutex_);
            tasks_.emplace([task] { (*task)(); });
        }
        ready_.notify_one();
        return completion;
    }

    void worker_loop_() {
        active_executor_ = this;
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock lock(mutex_);
                ready_.wait(lock,
                            [this] { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) break;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
        active_executor_ = nullptr;
    }

    inline static thread_local const ParallelExecutor* active_executor_ =
        nullptr;

    std::mutex mutex_;
    std::condition_variable ready_;
    std::queue<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
    bool stopping_ = false;
};

inline std::shared_ptr<ParallelExecutor> default_parallel_executor() {
    static auto executor = std::make_shared<ParallelExecutor>();
    return executor;
}

}  // namespace foretree
