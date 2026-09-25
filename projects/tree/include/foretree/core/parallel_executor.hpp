#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#    include <immintrin.h>
#endif

namespace foretree {

// Low-latency fork-join pool for `parallel_for`.
//
// One job runs at a time. The job's range is cut into chunks that the calling
// thread and the workers claim from an atomic counter, so the caller does work
// instead of blocking, and nothing is allocated per chunk. Workers spin briefly
// between jobs (tree training issues parallel_for calls back to back) before
// sleeping on an atomic wait. Nested calls from a worker, and a second caller
// while a job is running, execute inline. The first exception thrown by a
// chunk is rethrown in the caller.
class ParallelExecutor {
   public:
    explicit ParallelExecutor(
        unsigned thread_count = std::thread::hardware_concurrency()) {
        const unsigned count = std::max(1U, thread_count);
        workers_.reserve(count - 1);
        for (unsigned i = 1; i < count; ++i)
            workers_.emplace_back([this] { worker_loop_(); });
    }

    ParallelExecutor(const ParallelExecutor&) = delete;
    ParallelExecutor& operator=(const ParallelExecutor&) = delete;

    ~ParallelExecutor() {
        stopping_.store(true, std::memory_order_seq_cst);
        generation_.fetch_add(1, std::memory_order_seq_cst);
        generation_.notify_all();
        for (auto& worker : workers_)
            if (worker.joinable()) worker.join();
    }

    // Participating threads: the workers plus the calling thread.
    [[nodiscard]] unsigned thread_count() const noexcept {
        return static_cast<unsigned>(workers_.size() + 1);
    }

    template <class Function>
    void parallel_for(int begin, int end, int minimum_grain,
                      Function&& function) {
        const int count = end - begin;
        if (count <= 0) return;

        const int grain = std::max(1, minimum_grain);
        if (count <= grain || workers_.empty() || active_executor_ == this) {
            function(begin, end);
            return;
        }
        std::unique_lock submit(submit_mutex_, std::try_to_lock);
        if (!submit.owns_lock()) {  // another thread is running a job
            function(begin, end);
            return;
        }

        const int threads = static_cast<int>(thread_count());
        const int n_chunks =
            std::min((count + grain - 1) / grain, 4 * threads);
        const int chunk_size = (count + n_chunks - 1) / n_chunks;

        // Retire the previous job: invalidate its id, then wait until no
        // worker is between "joined" and "checked the id" (Dekker handshake
        // with the worker's active_ increment; both sides are seq_cst).
        job_id_.store(0, std::memory_order_seq_cst);
        while (active_.load(std::memory_order_seq_cst) != 0) pause_();

        using Fn = std::remove_reference_t<Function>;
        job_.context = const_cast<void*>(
            static_cast<const void*>(std::addressof(function)));
        job_.invoke = [](void* context, int b, int e) {
            (*static_cast<Fn*>(context))(b, e);
        };
        job_.begin = begin;
        job_.end = end;
        job_.chunk_size = chunk_size;
        job_.n_chunks = n_chunks;
        job_.next.store(0, std::memory_order_relaxed);
        job_.pending.store(n_chunks, std::memory_order_relaxed);
        job_.failure = nullptr;

        const uint64_t id = ++last_id_;
        job_id_.store(id, std::memory_order_seq_cst);
        generation_.store(id, std::memory_order_seq_cst);
        generation_.notify_all();

        // While the caller runs chunks it counts as inside this executor, so
        // a nested parallel_for from a chunk runs inline (it must not
        // re-lock submit_mutex_ on this thread).
        const ParallelExecutor* outer = std::exchange(active_executor_, this);
        run_chunks_();
        active_executor_ = outer;

        // Wait for chunks still running on workers.
        for (int spin = 0; job_.pending.load(std::memory_order_acquire) != 0;
             ++spin) {
            if (spin < kSpinIterations) {
                pause_();
            } else {
                const int pending = job_.pending.load(std::memory_order_acquire);
                if (pending != 0)
                    job_.pending.wait(pending, std::memory_order_acquire);
            }
        }
        if (job_.failure) {
            std::exception_ptr failure = std::exchange(job_.failure, nullptr);
            std::rethrow_exception(failure);
        }
    }

   private:
    static constexpr int kSpinIterations = 4000;

    struct Job {
        void* context = nullptr;
        void (*invoke)(void*, int, int) = nullptr;
        int begin = 0, end = 0, chunk_size = 1, n_chunks = 0;
        std::atomic<int> next{0};
        std::atomic<int> pending{0};
        std::exception_ptr failure;
        std::mutex failure_mutex;
    };

    static void pause_() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
        _mm_pause();
#else
        std::this_thread::yield();
#endif
    }

    // Claim and run chunks of the current job until none are left.
    void run_chunks_() {
        for (;;) {
            const int chunk = job_.next.fetch_add(1, std::memory_order_relaxed);
            if (chunk >= job_.n_chunks) return;
            const int b = job_.begin + chunk * job_.chunk_size;
            const int e = std::min(job_.end, b + job_.chunk_size);
            if (b < e) {
                try {
                    job_.invoke(job_.context, b, e);
                } catch (...) {
                    std::lock_guard lock(job_.failure_mutex);
                    if (!job_.failure) job_.failure = std::current_exception();
                }
            }
            if (job_.pending.fetch_sub(1, std::memory_order_acq_rel) == 1)
                job_.pending.notify_one();
        }
    }

    void worker_loop_() {
        active_executor_ = this;
        // Start from the initial generation (not a fresh load): a job or the
        // destructor may have bumped it before this thread got scheduled, and
        // loading it here would make the worker wait for a change that
        // already happened.
        uint64_t seen = 0;
        for (;;) {
            // Spin briefly for the next job, then sleep.
            uint64_t current = generation_.load(std::memory_order_acquire);
            for (int spin = 0; current == seen && spin < kSpinIterations;
                 ++spin) {
                pause_();
                current = generation_.load(std::memory_order_acquire);
            }
            if (current == seen) {
                generation_.wait(seen, std::memory_order_acquire);
                current = generation_.load(std::memory_order_acquire);
            }
            seen = current;
            if (stopping_.load(std::memory_order_acquire)) break;

            active_.fetch_add(1, std::memory_order_seq_cst);
            if (job_id_.load(std::memory_order_seq_cst) == current)
                run_chunks_();
            active_.fetch_sub(1, std::memory_order_seq_cst);
        }
        active_executor_ = nullptr;
    }

    inline static thread_local const ParallelExecutor* active_executor_ =
        nullptr;

    Job job_;
    std::atomic<uint64_t> generation_{0};
    std::atomic<uint64_t> job_id_{0};
    std::atomic<int> active_{0};
    std::atomic<bool> stopping_{false};
    uint64_t last_id_ = 0;
    std::mutex submit_mutex_;
    std::vector<std::thread> workers_;
};

inline std::shared_ptr<ParallelExecutor> default_parallel_executor() {
    static auto executor = std::make_shared<ParallelExecutor>();
    return executor;
}

}  // namespace foretree
